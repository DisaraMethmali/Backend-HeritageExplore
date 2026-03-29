"""
RAG Report Generator — rag_report_generator.py
================================================
Generates a per-user PDF report where each topic the user asked about
becomes one rich descriptive paragraph, built by combining:
  1. The actual RAG answers from the user's chat history
  2. Matching knowledge entries from data.json

Flask endpoints:
  GET /report/user?location=all|temple|galle
      Authorization: Bearer <token>   ← any logged-in user's token works
  GET /report/preview
      Authorization: Bearer <token>
"""

import io
import re
import json
import os
from datetime import datetime
from collections import defaultdict
from typing import Dict, List

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ── Colour palette ─────────────────────────────────────────────────────────────
C_NAVY          = colors.HexColor("#1E3A5F")
C_BLUE          = colors.HexColor("#2563EB")
C_LIGHT_BLUE    = colors.HexColor("#DBEAFE")
C_YELLOW        = colors.HexColor("#F5C518")
C_BLACK         = colors.HexColor("#111827")
C_DARK_GREY     = colors.HexColor("#374151")
C_MID_GREY      = colors.HexColor("#6B7280")
C_LIGHT_GREY    = colors.HexColor("#E5E7EB")
C_WHITE         = colors.white

# ── Topic constants ────────────────────────────────────────────────────────────
TEMPLE_TOPICS = {"temple", "buddhism", "festival", "king", "nilame"}
GALLE_TOPICS  = {"fort", "trade", "colonial", "dutch"}

TEMPLE_KEYWORDS = ["tooth relic", "dalada", "maligawa", "perahera", "buddhist",
                   "temple", "nilame", "ceremony", "kandy", "relic", "puja"]
GALLE_KEYWORDS  = ["galle", "fort", "dutch", "voc", "colonial", "cinnamon",
                   "bastion", "rampart", "trade", "ceylon coast", "warehouse"]

TOPIC_HEADINGS = {
    "temple":   "The Temple of the Sacred Tooth Relic",
    "buddhism": "Buddhism in Sri Lankan Heritage",
    "festival": "The Esala Perahera Festival",
    "king":     "The Kandyan Kings",
    "nilame":   "The Diyawadana Nilame",
    "fort":     "Galle Fort",
    "trade":    "The Spice Trade & Indian Ocean Commerce",
    "colonial": "Colonial Rule in Sri Lanka",
    "dutch":    "The Dutch VOC & Galle",
    "general":  "General Historical Context",
}

TOPIC_KEYWORDS_MAP = {
    "temple":   ["temple", "tooth relic", "dalada", "maligawa", "kandy", "shrine",
                 "casket", "relic", "puja", "sacred", "worship"],
    "buddhism": ["buddhism", "buddhist", "dhamma", "nirvana", "monk",
                 "sangha", "merit", "meditation"],
    "festival": ["perahera", "esala", "festival", "procession", "elephant",
                 "dancer", "drummer", "torch"],
    "king":     ["king", "royal", "ruler", "kingdom", "reign", "palace",
                 "rajasimha", "nayakkar"],
    "nilame":   ["nilame", "diyawadana", "custodian", "lay custodian",
                 "malvatta", "asgiriya"],
    "fort":     ["galle", "fort", "rampart", "bastion", "dutch",
                 "fortress", "wall", "harbour"],
    "trade":    ["trade", "spice", "cinnamon", "commerce", "merchant",
                 "port", "ocean", "ship"],
    "colonial": ["portuguese", "dutch", "british", "colonial", "occupation",
                 "invasion", "governor"],
    "dutch":    ["dutch", "voc", "holland", "netherlands"],
    "general":  ["sri lanka", "history", "ancient", "heritage", "culture",
                 "anuradhapura", "polonnaruva"],
}

# Maximum sentences to pull from the knowledge base per topic
_KB_MAX_SENTENCES = 8   # ← reduced to keep paragraphs page-safe


# ── Load data.json ─────────────────────────────────────────────────────────────

def _load_knowledge_base(path: str = "data.json") -> List[Dict]:
    if not os.path.exists(path):
        for candidate in ["data.json", "./data.json", "../data.json"]:
            if os.path.exists(candidate):
                path = candidate
                break
        else:
            return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _get_relevant_kb_sentences(topic: str, knowledge_base: List[Dict],
                                max_sentences: int = _KB_MAX_SENTENCES) -> List[str]:
    keywords = TOPIC_KEYWORDS_MAP.get(topic, [topic])
    scored   = []
    for entry in knowledge_base:
        instruction = (entry.get("instruction") or "").lower()
        output      = (entry.get("output") or "").strip()
        if not output:
            continue
        score = sum(1 for kw in keywords if kw in instruction or kw in output.lower())
        if score > 0:
            scored.append((score, output))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:max_sentences]]


# ── Text helpers ───────────────────────────────────────────────────────────────

def _classify_message(record: Dict) -> str:
    topic   = (record.get("topic") or "").lower()
    char_id = (record.get("character_id") or "").lower()
    text    = (record.get("question") or "").lower() + " " + (record.get("answer") or "").lower()

    temple_score = galle_score = 0
    if topic in TEMPLE_TOPICS:        temple_score += 3
    if topic in GALLE_TOPICS:         galle_score  += 3
    if char_id in ("king", "nilame"): temple_score += 2
    if char_id == "dutch":            galle_score  += 2
    for kw in TEMPLE_KEYWORDS:
        if kw in text: temple_score += 1
    for kw in GALLE_KEYWORDS:
        if kw in text: galle_score  += 1

    if temple_score == 0 and galle_score == 0:
        return "general"
    return "temple" if temple_score >= galle_score else "galle"


def _deduplicate_sentences(sentences: List[str]) -> List[str]:
    def _words(text):
        return set(re.sub(r'[^a-z0-9 ]', '', text.lower()).split())

    kept      = []
    kept_sets = []
    for s in sentences:
        ws = _words(s)
        if not ws:
            continue
        duplicate = any(
            len(ws & ks) / min(len(ws), len(ks)) > 0.65
            for ks in kept_sets if ks
        )
        if not duplicate:
            kept.append(s)
            kept_sets.append(ws)
    return kept


# Approximate page body height in points (A4 minus margins)
_PAGE_BODY_PT = (297 - 15 - 18) * mm   # ~736 pt
# Conservative max chars per page at 10.5pt / leading 18 (~72 chars/line, ~40 lines)
_MAX_CHARS = 2_200


def _build_paragraph(topic: str,
                     chat_records: List[Dict],
                     knowledge_base: List[Dict]) -> str:
    all_sentences = []

    for r in chat_records:
        answer = (r.get("answer") or "").strip()
        if answer:
            for s in re.split(r'(?<=[.!?])\s+', answer):
                s = s.strip()
                if len(s) > 25:
                    all_sentences.append(s)

    kb_sentences = _get_relevant_kb_sentences(topic, knowledge_base)
    for output in kb_sentences:
        for s in re.split(r'(?<=[.!?])\s+', output):
            s = s.strip()
            if len(s) > 25:
                all_sentences.append(s)

    if not all_sentences:
        return "No detailed information was available for this topic."

    unique  = _deduplicate_sentences(all_sentences)
    cleaned = []
    total   = 0
    for s in unique:
        if s[-1] not in ".!?":
            s += "."
        if total + len(s) + 1 > _MAX_CHARS:   # ← hard cap to prevent overflow
            break
        cleaned.append(s)
        total += len(s) + 1

    return " ".join(cleaned)


# ── Styles ─────────────────────────────────────────────────────────────────────

def _build_styles():
    base   = getSampleStyleSheet()
    styles = {}

    styles["report_title"] = ParagraphStyle(
        "report_title", parent=base["Title"],
        fontSize=26, textColor=C_WHITE, alignment=TA_CENTER,
        spaceAfter=2, spaceBefore=0, fontName="Helvetica-Bold", leading=32,
    )
    styles["report_sub"] = ParagraphStyle(
        "report_sub", parent=base["Normal"],
        fontSize=10, textColor=C_YELLOW, alignment=TA_CENTER,
        spaceAfter=0, spaceBefore=0, fontName="Helvetica", leading=14,
    )
    styles["topic_banner"] = ParagraphStyle(
        "topic_banner", parent=base["Heading1"],
        fontSize=13, textColor=C_WHITE, fontName="Helvetica-Bold",
        alignment=TA_LEFT, spaceAfter=0, spaceBefore=0, leading=18,
    )
    styles["topic_number"] = ParagraphStyle(
        "topic_number", parent=base["Normal"],
        fontSize=13, textColor=C_YELLOW, fontName="Helvetica-Bold",
        alignment=TA_LEFT, spaceAfter=0, spaceBefore=0, leading=18,
    )
    styles["body"] = ParagraphStyle(
        "body", parent=base["Normal"],
        fontSize=10.5, textColor=C_BLACK,
        leading=18, spaceAfter=0, spaceBefore=8,
        firstLineIndent=0, fontName="Helvetica",
        leftIndent=4, rightIndent=4,
    )
    styles["footer"] = ParagraphStyle(
        "footer", parent=base["Normal"],
        fontSize=8, textColor=C_MID_GREY,
        alignment=TA_CENTER, fontName="Helvetica", leading=11,
    )
    styles["footer_right"] = ParagraphStyle(
        "footer_right", parent=base["Normal"],
        fontSize=8, textColor=C_MID_GREY,
        alignment=TA_RIGHT, fontName="Helvetica", leading=11,
    )
    return styles


# ── Report sections ────────────────────────────────────────────────────────────

def _title_section(story, styles, generated_at: str):
    title_table = Table(
        [[Paragraph("Heritage Report", styles["report_title"])]],
        colWidths=[166 * mm],
    )
    title_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
        ("LEFTPADDING",   (0, 0), (-1, -1), 18),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 18),
    ]))
    story.append(title_table)

    accent = Table(
        [[Paragraph("Historical Knowledge Summary", styles["report_sub"])]],
        colWidths=[166 * mm],
    )
    accent.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 18),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 18),
    ]))
    story.append(accent)
    story.append(Spacer(1, 8 * mm))


def _topic_section(story, styles, topic: str, index: int,
                   chat_records: List[Dict],
                   knowledge_base: List[Dict]):
    heading = TOPIC_HEADINGS.get(topic, topic.title())
    number  = f"{index:02d}"

    # Blue section banner
    banner = Table(
        [[
            Paragraph(number, styles["topic_number"]),
            Paragraph(heading, styles["topic_banner"]),
        ]],
        colWidths=[14 * mm, 152 * mm],
    )
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (0, 0),   12),
        ("LEFTPADDING",   (1, 0), (1, 0),   6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))

    # Yellow accent rule
    rule = HRFlowable(
        width="100%", thickness=3, color=C_YELLOW,
        spaceAfter=0, spaceBefore=0
    )

    # Body: plain Paragraph — no box, no table wrapper, flows freely across pages
    paragraph_text = _build_paragraph(topic, chat_records, knowledge_base)
    body_para      = Paragraph(paragraph_text, styles["body"])

    # Keep banner + rule together; body flows naturally (no box, no overflow risk)
    story.append(KeepTogether([banner, rule]))
    story.append(body_para)
    story.append(Spacer(1, 6 * mm))


def _footer_section(story, styles, generated_at: str):
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(
        width="100%", thickness=1.5, color=C_YELLOW,
        spaceAfter=4, spaceBefore=0
    ))
    date_str = generated_at[:16].replace("T", "  ") + "  UTC"
    footer = Table(
        [[
            Paragraph("Heritage Report  ·  Confidential", styles["footer"]),
            Paragraph(f"Generated: {date_str}", styles["footer_right"]),
        ]],
        colWidths=[83 * mm, 83 * mm],
    )
    footer.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_LIGHT_BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(footer)


# ── Main public function ───────────────────────────────────────────────────────

def generate_user_report(
    username:            str,
    full_name:           str,
    records:             List[Dict],
    expertise_level:     str = "tourist",
    knowledge_base_path: str = "data.json",
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=15 * mm, bottomMargin=18 * mm,
        title="Heritage Report",
        author="", subject="Heritage Report",
        allowSplitting=1,      # ← ensure ReportLab can split flowables
    )

    styles         = _build_styles()
    story          = []
    generated      = datetime.utcnow().isoformat()
    knowledge_base = _load_knowledge_base(knowledge_base_path)

    topic_groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        topic = (r.get("topic") or "general").lower().strip()
        topic_groups[topic].append(r)

    def _order(t):
        if t in TEMPLE_TOPICS: return 0
        if t in GALLE_TOPICS:  return 1
        return 2

    sorted_topics = sorted(topic_groups.keys(), key=_order)

    _title_section(story, styles, generated)
    for i, topic in enumerate(sorted_topics):
        if i > 0 and i % 4 == 0:
            story.append(PageBreak())
        _topic_section(story, styles, topic, i + 1, topic_groups[topic], knowledge_base)
    _footer_section(story, styles, generated)

    doc.build(story)
    return buf.getvalue()


# ── Flask route registration ───────────────────────────────────────────────────

def register_report_route(app, chatbot,
                           knowledge_base_path: str = "data.json"):
    from flask import request, send_file, jsonify
    import io as _io

    def _resolve_token(req):
        auth_header = req.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        else:
            token = req.args.get("token", "").strip()
        if not token:
            return None, None
        if not chatbot.auth:
            return None, None
        session_info = chatbot.auth.verify_token(token)
        return token, session_info

    @app.route("/report/user", methods=["GET"])
    def user_report():
        token, session_info = _resolve_token(request)
        if not session_info:
            return jsonify({
                "error":  "Unauthorized — please log in first.",
                "how_to": (
                    "1. POST /auth/login with your username & password to get a token. "
                    "2. Use: Authorization: Bearer <token>  OR  ?token=<token>"
                )
            }), 401

        username  = session_info["username"]
        user_data = chatbot.auth.get_user_profile(username) or {}
        full_name = user_data.get("full_name") or username
        expertise = user_data.get("expertise_level", "tourist")

        if not chatbot.history_mgr:
            return jsonify({"error": "History manager not available"}), 503

        all_records     = chatbot.history_mgr.export_history(username)
        location_filter = request.args.get("location", "all").lower()

        if location_filter == "temple":
            records  = [r for r in all_records if _classify_message(r) == "temple"]
            filename = f"temple_report_{username}.pdf"
        elif location_filter == "galle":
            records  = [r for r in all_records if _classify_message(r) == "galle"]
            filename = f"galle_report_{username}.pdf"
        else:
            records  = all_records
            filename = f"full_report_{username}.pdf"

        if not records:
            return jsonify({
                "error":    "No chat history found for your account.",
                "username": username,
                "hint":     "Start chatting with the historical characters first.",
            }), 404

        try:
            pdf_bytes = generate_user_report(
                username, full_name, records, expertise,
                knowledge_base_path=knowledge_base_path
            )
            return send_file(
                _io.BytesIO(pdf_bytes),
                mimetype="application/pdf",
                as_attachment=True,
                download_name=filename,
            )
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/report/preview", methods=["GET"])
    def report_preview():
        token, session_info = _resolve_token(request)
        if not session_info:
            return jsonify({
                "error":  "Unauthorized — please log in first.",
                "how_to": "Authorization: Bearer <token>  or  ?token=<token>"
            }), 401

        username = session_info["username"]
        if not chatbot.history_mgr:
            return jsonify({"error": "History manager not available"}), 503

        records      = chatbot.history_mgr.export_history(username)
        topic_groups = defaultdict(int)
        for r in records:
            topic_groups[(r.get("topic") or "general").lower()] += 1

        return jsonify({
            "success":         True,
            "username":        username,
            "topics_explored": dict(topic_groups),
            "total_messages":  len(records),
            "download_urls": {
                "full":   f"/report/user?location=all&token={token}",
                "temple": f"/report/user?location=temple&token={token}",
                "galle":  f"/report/user?location=galle&token={token}",
            },
            "timestamp": datetime.utcnow().isoformat(),
        })
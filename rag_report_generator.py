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

CHANGES FROM ORIGINAL:
  - /report/preview now returns rich topic_summaries with key_points,
    has_temple_report, has_galle_report, characters_used, unique_sessions,
    avg_confidence_pct, temple_messages, galle_messages, general_messages.
  - Two new helper functions added: _classify_topic(), _build_topic_summary()
  - All existing fields (topics_explored, download_urls) are preserved for
    backward compatibility with the Gradio report tab.
"""

import io
import re
import json
import os
import difflib
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
C_NAVY          = colors.HexColor("#0D1B2A")
C_BLUE          = colors.HexColor("#1A3A5C")
C_BLUE_MID      = colors.HexColor("#1E6091")
C_BLUE_LIGHT    = colors.HexColor("#2196A6")
C_GOLD          = colors.HexColor("#C9A84C")
C_GOLD_LIGHT    = colors.HexColor("#F5D98B")
C_CREAM         = colors.HexColor("#FDF6E3")
C_CREAM_DARK    = colors.HexColor("#F5EDD6")
C_BLACK         = colors.HexColor("#0D1B2A")
C_DARK_GREY     = colors.HexColor("#2C3E50")
C_MID_GREY      = colors.HexColor("#5D6D7E")
C_LIGHT_GREY    = colors.HexColor("#ECF0F1")
C_WHITE         = colors.white
C_TEMPLE        = colors.HexColor("#6B21A8")   # purple for temple topics
C_GALLE         = colors.HexColor("#1D4ED8")   # blue for galle topics
C_ACCENT        = colors.HexColor("#C9A84C")   # gold accent

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

_KB_MAX_SENTENCES = 8

# ── Character name maps ───────────────────────────────────────────────────────
_CHARACTER_NAMES = {
    "king":    "King Sri Wikrama Rajasinha",
    "nilame":  "the Diyawadana Nilame",
    "dutch":   "Captain Willem van der Berg",
    "citizen": "Sunil",
}

_FIRST_PERSON_SUBS = [
    # Longer phrases first
    (r"\bI have the sacred honor\b",    "{name} has the sacred honor"),
    (r"\bI have dedicated\b",           "{name} has dedicated"),
    (r"\bI have spent\b",               "{name} has spent"),
    (r"\bI have the honor\b",           "{name} has the honor"),
    (r"\bI am the\b",                   "{name} is the"),
    (r"\bI am responsible\b",           "{name} is responsible"),
    (r"\bI am tasked\b",                "{name} is tasked"),
    (r"\bI ruled\b",                    "{name} ruled"),
    (r"\bI served\b",                   "{name} served"),
    (r"\bI controlled\b",               "{name} controlled"),
    (r"\bI built\b",                    "{name} built"),
    (r"\bI established\b",              "{name} established"),
    (r"\bI dedicated\b",                "{name} dedicated"),
    (r"\bI focused\b",                  "{name} focused"),
    (r"\bI protected\b",                "{name} protected"),
    (r"\bI managed\b",                  "{name} managed"),
    (r"\bI organized\b",                "{name} organized"),
    (r"\bI oversaw\b",                  "{name} oversaw"),
    (r"\bI performed\b",                "{name} performed"),
    (r"\bI conducted\b",                "{name} conducted"),
    (r"\bMy reign\b",                   "{name}'s reign"),
    (r"\bmy reign\b",                   "{name}'s reign"),
    (r"\bMy kingdom\b",                 "{name}'s kingdom"),
    (r"\bmy kingdom\b",                 "{name}'s kingdom"),
    (r"\bMy sacred duties\b",           "{name}'s sacred duties"),
    (r"\bmy sacred duties\b",           "{name}'s sacred duties"),
    (r"\bMy duties\b",                  "{name}'s duties"),
    (r"\bmy duties\b",                  "{name}'s duties"),
    (r"\bour kingdom\b",                "the Kingdom of Kandy"),
    (r"\bOur kingdom\b",                "the Kingdom of Kandy"),
    (r"\bour Buddhist traditions\b",    "the Buddhist traditions"),
    (r"\bOur Buddhist traditions\b",    "the Buddhist traditions"),
]

# ── Hardcoded spelling fixes (fast, runs first) ───────────────────────────────
_SPELLING_FIXES = [
    (r"\bGalle Port\b",  "Galle Fort"),
    (r"\bgalle port\b",  "Galle Fort"),
    (r"\bGalle port\b",  "Galle Fort"),
    (r"\bport\b", "Fort"),
]

# ── Known correct terms for fuzzy matching ────────────────────────────────────
_KNOWN_CORRECT_TERMS = [
    "Galle Fort", "Sacred Tooth Relic", "Kingdom of Kandy",
    "Esala Perahera", "Sri Dalada Maligawa", "Dutch East India Company",
    "Sri Wikrama Rajasinha", "Diyawadana Nilame", "Buddhism",
    "Anuradhapura", "Polonnaruwa", "VOC",
]

def _fix_spelling(text: str) -> str:
    # Step 1 — apply hardcoded fixes first (fast, exact)
    for pattern, replacement in _SPELLING_FIXES:
        text = re.sub(pattern, replacement, text)

    # Step 2 — fuzzy-fix multi-word terms using difflib
    words = text.split()
    for i in range(len(words)):
        for term in _KNOWN_CORRECT_TERMS:
            term_words = term.split()
            n          = len(term_words)
            chunk      = " ".join(words[i:i + n])
            if not chunk:
                continue
            ratio = difflib.SequenceMatcher(
                None, chunk.lower(), term.lower()
            ).ratio()
            # 0.80 = close enough to correct, but not already exact
            if 0.80 <= ratio < 1.0:
                text = text.replace(chunk, term, 1)
                # Rebuild word list after replacement so indexes stay valid
                words = text.split()
                break
    return text

def _substitute_character_name(sentence: str, character_id: str) -> str:
    """Replace first-person pronouns with the character's actual name."""
    name = _CHARACTER_NAMES.get(character_id, "")
    if not name:
        return sentence
    result = sentence
    for pattern, template in _FIRST_PERSON_SUBS:
        replacement = template.format(name=name)
        result = re.sub(pattern, replacement, result)
    return result


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
    """
    UNCHANGED — existing function used by /report/user for PDF filtering.
    Kept exactly as-is for backward compatibility.
    """
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


# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 1 — NEW helper: _classify_topic()
# Used by the enriched /report/preview to bucket records into
# temple | galle | general using a fast keyword scan.
# Note: _classify_message() above is the heavier weighted scorer used for
# PDF generation. _classify_topic() is the lightweight version for preview.
# ══════════════════════════════════════════════════════════════════════════════

_TEMPLE_KW = {"temple", "dalada", "maligawa", "tooth", "relic", "perahera",
               "kandy", "nilame", "king", "buddhis", "festival", "ceremony"}
_GALLE_KW  = {"galle", "fort", "dutch", "voc", "bastion", "rampart", "colonial",
               "maritime", "trade", "cinnamon", "portuguese"}


def _classify_topic(topic: str, question: str = "", answer: str = "") -> str:
    """Classify a single record into temple | galle | general."""
    text = (topic + " " + question + " " + answer).lower()
    ts = sum(1 for w in _TEMPLE_KW if w in text)
    gs = sum(1 for w in _GALLE_KW  if w in text)
    if ts > gs:   return "temple"
    if gs > ts:   return "galle"
    return "general"




# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 2 — NEW helper: _build_topic_summary()
# Extracts up to 5 deduplicated key points from a list of chat records
# belonging to the same location bucket (temple/galle/general).
# Called only by the enriched /report/preview route below.
# ══════════════════════════════════════════════════════════════════════════════

# ── Junk sentence filters ─────────────────────────────────────────────────────
_JUNK_PATTERNS = [
    r"^(ayubowan|hello|hi|sure|sadhu|goedendag|ok|yes|no)[!.,\s]",
    r"^(i am|i'm|i apologize|thank you|here is|here are|in response|as requested)",
    r"^(response:|history question:|note:|update:|regarding)",
    r"^(after|based on) (the|your|this) (previous|feedback|question|update)",
    r"would you like",
    r"please (ask|feel free|let me)",
    r"ask me anything",
    r"how (may|can) i (assist|help)",
    r"^please provide",
    r"^here's how",
    r"^here is how",
    r"^the person referred to as",
    r"^is not mentioned",
    r"^i've spent my life",
    r"^i have spent my life",
    r"^provide another example",
    r"^give me another",
    r"additional information based on",
    r"^for the same passage",
    r"^based on the given",
    r"^according to the given",
    r"^from the given material",
    r"given (historical )?knowledge material",
    r"^i don't have specific information",
    r"^i apologize",
    r"cannot be guaranteed",
    r"^it's clear that",
    r"^overall,",
    r"^in conclusion,",
    r"^i've updated",
    r"^i have updated",
    r"^i've revised",
    r"^i have revised",
    r"^i've added",
    r"^i have added",
    r"^thank you for your feedback",
    r"^thank you for (the|your)",
    r"^as per your",
    r"^as per the",
    r"^i've incorporated",
    r"^i have incorporated",
    r"^i've included",
    r"^i have included",
    r"^regarding (the|your) (previous|last|earlier)",
    r"^in response to your",
    r"^to answer your",
    r"^to address your",
    r"historical knowledge section",
    r"previous(ly)? (asked|mentioned|provided|given)",
    r"^based on your previous",
    r"^following your",
]
_JUNK_RE = [re.compile(p, re.IGNORECASE) for p in _JUNK_PATTERNS]

_MIN_HISTORICAL_WORDS = 10   # minimum words for a sentence to be "real" content

def _is_historical_sentence(sentence: str) -> bool:
    """
    Returns True only if the sentence looks like genuine historical content.
    Rejects greetings, meta-text, boilerplate, and very short sentences.
    """
    s = sentence.strip()
    if len(s.split()) < _MIN_HISTORICAL_WORDS:
        return False
    for pattern in _JUNK_RE:
        if pattern.search(s):
            return False
    return True

def _is_duplicate(sentence: str, seen_sentences: list) -> bool:
    """
    Returns True if sentence is too similar to any already-seen sentence.
    Uses both prefix check (fast) and full similarity check (accurate).
    """
    key = sentence[:50]
    # Fast prefix check
    if key in {s[:50] for s in seen_sentences}:
        return True
    # Similarity check for near-duplicates
    for existing in seen_sentences:
        ratio = difflib.SequenceMatcher(
            None,
            sentence.lower(),
            existing.lower()
        ).ratio()
        if ratio > 0.75:  # 75% similar = treat as duplicate
            return True
    return False


def _build_topic_summary(records: list) -> dict:
    """
    Given chat records for one location bucket, return:
      { key_points: [str, ...],  summary: str }

    key_points — best historical sentences extracted from answers, max 6.
    summary    — 1-sentence human-readable description.
    """
    seen_sentences = []   # change from set to list
    key_points = []
    max_points = _dynamic_max_points(len(records))

    for r in records:
        ans     = (r.get("answer") or "").strip()
        char_id = (r.get("character_id") or "").lower()
        if not ans:
            continue

        # Split into individual sentences
        sentences = re.split(r'(?<=[.!?])\s+', ans)
        for s in sentences:
            s = s.strip()
            if not s or not _is_historical_sentence(s): # Apply quality filter
                continue
            # Apply spelling fix then character name substitution
            s = _fix_spelling(s)
            s = _substitute_character_name(s, char_id)

            if _is_duplicate(s, seen_sentences):  # smarter check
                continue

            seen_sentences.append(s)
            key_points.append(s)
            if len(key_points) >= max_points:
                break
        if len(key_points) >= max_points:
            break

    chars   = list({r.get("character_id", "?") for r in records if r.get("character_id")})
    summary = (
        f"Covered {len(records)} conversation{'s' if len(records) != 1 else ''} "
        f"with {', '.join(chars) if chars else 'historical characters'}."
    )
    return {"key_points": key_points, "summary": summary}




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


_PAGE_BODY_PT = (297 - 15 - 18) * mm
_MAX_CHARS = 2_200

def _dynamic_max_points(record_count: int) -> int:
    """
    Scale the number of key points based on conversation count.
    Short chats get fewer points, long chats get more.
    """
    if record_count <= 3:
        return 4
    elif record_count <= 8:
        return 5
    elif record_count <= 20:
        return 7
    elif record_count <= 40:
        return 9
    else:
        return 12
    


def _build_key_points_for_pdf(topic: str,
                               chat_records: List[Dict],
                               knowledge_base: List[Dict],
                               max_points: int = 8) -> List[str]:
    """
    Extract clean historical key points for PDF rendering.
    Same logic as _build_topic_summary but allows more points
    and also pulls from the knowledge base.
    """
    seen_sentences = []   # changed from set to list for _is_duplicate()
    key_points = []

    # From chat history first
    for r in chat_records:
        ans     = (r.get("answer") or "").strip()
        char_id = (r.get("character_id") or "").lower()
        if not ans:
            continue
        for s in re.split(r'(?<=[.!?])\s+', ans):
            s = s.strip()
            if not s or not _is_historical_sentence(s):
                continue
            s = _fix_spelling(s)
            s = _substitute_character_name(s, char_id)
            if _is_duplicate(s, seen_sentences):   # smarter check
                continue
            seen_sentences.append(s)
            key_points.append(s)
            if len(key_points) >= max_points:
                return key_points

    # Fill remaining from knowledge base
    kb_sentences = _get_relevant_kb_sentences(topic, knowledge_base, max_sentences=10)
    for output in kb_sentences:
        for s in re.split(r'(?<=[.!?])\s+', output):
            s = s.strip()
            if not s or not _is_historical_sentence(s):
                continue
            s = _fix_spelling(s)
            if _is_duplicate(s, seen_sentences):   # smarter check
                continue
            seen_sentences.append(s)
            key_points.append(s)
            if len(key_points) >= max_points:
                return key_points

    return key_points

# Topics to EXCLUDE from the PDF report
_EXCLUDED_PDF_TOPICS = {"general", "kingdom"}


# ── Styles ─────────────────────────────────────────────────────────────────────
def _topic_accent_color(topic: str):
    """Return a unique accent color per topic category."""
    if topic in TEMPLE_TOPICS:
        return C_TEMPLE
    if topic in GALLE_TOPICS:
        return C_GALLE
    return C_BLUE_MID


def _build_styles():
    base   = getSampleStyleSheet()
    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title", parent=base["Title"],
        fontSize=32, textColor=C_GOLD,
        alignment=TA_CENTER,
        spaceAfter=4, spaceBefore=0,
        fontName="Helvetica-Bold", leading=38,
    )
    styles["cover_sub"] = ParagraphStyle(
        "cover_sub", parent=base["Normal"],
        fontSize=12, textColor=C_WHITE,
        alignment=TA_CENTER,
        spaceAfter=0, spaceBefore=0,
        fontName="Helvetica", leading=16,
    )
    styles["cover_user"] = ParagraphStyle(
        "cover_user", parent=base["Normal"],
        fontSize=11, textColor=C_GOLD_LIGHT,
        alignment=TA_CENTER,
        spaceAfter=0, spaceBefore=0,
        fontName="Helvetica-Bold", leading=14,
    )
    styles["cover_date"] = ParagraphStyle(
        "cover_date", parent=base["Normal"],
        fontSize=9, textColor=C_LIGHT_GREY,
        alignment=TA_CENTER,
        spaceAfter=0, spaceBefore=0,
        fontName="Helvetica", leading=12,
    )
    styles["topic_label"] = ParagraphStyle(
        "topic_label", parent=base["Normal"],
        fontSize=9, textColor=C_WHITE,
        fontName="Helvetica-Bold",
        alignment=TA_LEFT,
        spaceAfter=0, spaceBefore=0,
        leading=12, letterSpacing=1.5,
    )
    styles["topic_heading"] = ParagraphStyle(
        "topic_heading", parent=base["Heading1"],
        fontSize=15, textColor=C_WHITE,
        fontName="Helvetica-Bold",
        alignment=TA_LEFT,
        spaceAfter=0, spaceBefore=0, leading=20,
    )
    styles["bullet_point"] = ParagraphStyle(
        "bullet_point", parent=base["Normal"],
        fontSize=10.5, textColor=C_BLACK,
        leading=19, spaceAfter=0, spaceBefore=0,
        leftIndent=0, firstLineIndent=0,
        fontName="Helvetica",
        rightIndent=0,
    )
    styles["bullet_point_alt"] = ParagraphStyle(
        "bullet_point_alt", parent=base["Normal"],
        fontSize=10.5, textColor=C_DARK_GREY,
        leading=19, spaceAfter=0, spaceBefore=0,
        leftIndent=0, firstLineIndent=0,
        fontName="Helvetica",
        rightIndent=0,
    )
    styles["no_data"] = ParagraphStyle(
        "no_data", parent=base["Normal"],
        fontSize=10, textColor=C_MID_GREY,
        leading=16, spaceAfter=0, spaceBefore=6,
        leftIndent=6, fontName="Helvetica-Oblique",
    )
    styles["footer_left"] = ParagraphStyle(
        "footer_left", parent=base["Normal"],
        fontSize=8, textColor=C_MID_GREY,
        alignment=TA_LEFT,
        fontName="Helvetica", leading=10,
    )
    styles["footer_right"] = ParagraphStyle(
        "footer_right", parent=base["Normal"],
        fontSize=8, textColor=C_MID_GREY,
        alignment=TA_RIGHT,
        fontName="Helvetica", leading=10,
    )
    styles["section_intro"] = ParagraphStyle(
        "section_intro", parent=base["Normal"],
        fontSize=9.5, textColor=C_MID_GREY,
        leading=14, spaceAfter=0, spaceBefore=4,
        leftIndent=4, fontName="Helvetica-Oblique",
    )
    return styles


# ── Report sections ────────────────────────────────────────────────────────────

def _title_section(story, styles, generated_at: str, full_name: str = ""):
    """
    Full-page cover — fills entire A4 page with deep navy background,
    gold title, decorative elements, user name, date, location legend.
    """
    page_w = A4[0] - 44 * mm   # usable width
    page_h = A4[1] - 33 * mm   # usable height (full page minus margins)

    # ── Full page cover table ─────────────────────────────────────────────────
    date_str = generated_at[:10]

    cover_rows = [
        # Top spacer
        [Spacer(1, 18 * mm)],

        # Decorative top rule
        [HRFlowable(width="40%", thickness=1.5, color=C_GOLD_LIGHT,
                    spaceAfter=0, spaceBefore=0)],

        [Spacer(1, 8 * mm)],

        # Small icon label
        [Paragraph(
            "SRI LANKA HERITAGE",
            ParagraphStyle("icon_label", fontSize=9, textColor=C_GOLD_LIGHT,
                           alignment=TA_CENTER, fontName="Helvetica",
                           leading=12, letterSpacing=3,
                           parent=getSampleStyleSheet()["Normal"])
        )],

        [Spacer(1, 6 * mm)],

        # Main title
        [Paragraph(
            "HISTORICAL",
            ParagraphStyle("main_t1", fontSize=38, textColor=C_GOLD,
                           alignment=TA_CENTER, fontName="Helvetica-Bold",
                           leading=44, parent=getSampleStyleSheet()["Normal"])
        )],
        [Paragraph(
            "JOURNEY",
            ParagraphStyle("main_t2", fontSize=38, textColor=C_GOLD,
                           alignment=TA_CENTER, fontName="Helvetica-Bold",
                           leading=44, parent=getSampleStyleSheet()["Normal"])
        )],

        [Spacer(1, 4 * mm)],

        # Subtitle
        [Paragraph(
            "HERITAGE&nbsp;&nbsp;REPORT",
            ParagraphStyle("sub_t", fontSize=11, textColor=C_GOLD_LIGHT,
                           alignment=TA_CENTER, fontName="Helvetica",
                           leading=16, parent=getSampleStyleSheet()["Normal"])
        )],

        [Spacer(1, 10 * mm)],

        # Decorative middle divider
        [HRFlowable(width="55%", thickness=0.8, color=C_GOLD_LIGHT,
                    spaceAfter=0, spaceBefore=0)],

        [Spacer(1, 10 * mm)],

        # Description
        [Paragraph(
            "A personalised summary of your conversations with<br/>"
            "Sri Lanka's historical characters",
            ParagraphStyle("desc", fontSize=11, textColor=C_WHITE,
                           alignment=TA_CENTER, fontName="Helvetica",
                           leading=18, parent=getSampleStyleSheet()["Normal"])
        )],

        [Spacer(1, 14 * mm)],
    ]

    # Prepared for
    if full_name:
        cover_rows += [
            [HRFlowable(width="30%", thickness=0.5,
                        color=C_GOLD_LIGHT, spaceAfter=0, spaceBefore=0)],
            [Spacer(1, 5 * mm)],
            [Paragraph(
                "PREPARED FOR",
                ParagraphStyle("pf_label", fontSize=8, textColor=C_GOLD_LIGHT,
                               alignment=TA_CENTER, fontName="Helvetica",
                               leading=11, letterSpacing=2,
                               parent=getSampleStyleSheet()["Normal"])
            )],
            [Spacer(1, 3 * mm)],
            [Paragraph(
                full_name,
                ParagraphStyle("pf_name", fontSize=16, textColor=C_GOLD,
                               alignment=TA_CENTER, fontName="Helvetica-Bold",
                               leading=20, parent=getSampleStyleSheet()["Normal"])
            )],
            [Spacer(1, 5 * mm)],
            [HRFlowable(width="30%", thickness=0.5,
                        color=C_GOLD_LIGHT, spaceAfter=0, spaceBefore=0)],
            [Spacer(1, 10 * mm)],
        ]

    # Generated date
    cover_rows += [
        [Paragraph(
            f"Generated  {date_str}",
            ParagraphStyle("gen_date", fontSize=9, textColor=C_MID_GREY,
                           alignment=TA_CENTER, fontName="Helvetica",
                           leading=12, parent=getSampleStyleSheet()["Normal"])
        )],
        # Bottom spacer fills remaining space
        [Spacer(1, 20 * mm)],
    ]

    cover_table = Table(cover_rows, colWidths=[page_w])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 8 * mm))

    # ── Location legend ───────────────────────────────────
    legend_data = [[
        Paragraph(
            "TEMPLE OF THE SACRED TOOTH  •  KANDY",
            ParagraphStyle("leg1", fontSize=9, textColor=C_TEMPLE,
                           fontName="Helvetica-Bold", alignment=TA_CENTER,
                           parent=getSampleStyleSheet()["Normal"])
        ),
        Paragraph(
            "GALLE FORT  •  SOUTHERN COAST",
            ParagraphStyle("leg2", fontSize=9, textColor=C_GALLE,
                           fontName="Helvetica-Bold", alignment=TA_CENTER,
                           parent=getSampleStyleSheet()["Normal"])
        ),
    ]]
    legend_table = Table(legend_data, colWidths=[page_w / 2, page_w / 2])
    legend_table.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("BOX",           (0, 0), (0, 0), 0.8, C_TEMPLE),
        ("BOX",           (1, 0), (1, 0), 0.8, C_GALLE),
        ("BACKGROUND",    (0, 0), (0, 0),
         colors.HexColor("#6B21A8").clone(alpha=0.06)),
        ("BACKGROUND",    (1, 0), (1, 0),
         colors.HexColor("#1D4ED8").clone(alpha=0.06)),
    ]))
    story.append(legend_table)

    # ── Force page break after cover ──────────────────────────────────────────
    story.append(PageBreak())   


def _topic_section(story, styles, topic: str, index: int,
                   chat_records: List[Dict],
                   knowledge_base: List[Dict],
                   max_points: int = 8):

    heading      = TOPIC_HEADINGS.get(topic, topic.title())
    accent_color = _topic_accent_color(topic)
    page_w       = A4[0] - 44 * mm

    # ── Category label (TEMPLE / GALLE FORT) ─────────────────────────────────
    if topic in TEMPLE_TOPICS:
        cat_label = "TEMPLE OF THE SACRED TOOTH  •  KANDY"
    elif topic in GALLE_TOPICS:
        cat_label = "GALLE FORT  •  SOUTHERN COAST"
    else:
        cat_label = "SRI LANKAN HERITAGE"

    # ── Header block ──────────────────────────────────────────────────────────
    num_style = ParagraphStyle(
        "num", fontSize=22, textColor=C_GOLD,
        fontName="Helvetica-Bold", alignment=TA_CENTER,
        leading=28, parent=getSampleStyleSheet()["Normal"]
    )
    header_data = [[
        Paragraph(f"{index:02d}", num_style),
        [
            Paragraph(cat_label.upper(), styles["topic_label"]),
            Paragraph(heading, styles["topic_heading"]),
        ],
    ]]
    header_table = Table(
        header_data,
        colWidths=[18 * mm, page_w - 18 * mm],
    )
    header_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), accent_color),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING",   (0, 0), (0, 0),   10),
        ("LEFTPADDING",   (1, 0), (1, 0),   10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))

    # ── Gold accent rule ──────────────────────────────────────────────────────
    rule = HRFlowable(
        width="100%", thickness=3, color=C_GOLD,
        spaceAfter=6, spaceBefore=0
    )

    # ── Key points ────────────────────────────────────────────────────────────
    key_points = _build_key_points_for_pdf(
        topic, chat_records, knowledge_base, max_points=max_points
    )

    flowables = [header_table, rule]

    if key_points:
        # Alternating background rows for readability
        for i, point in enumerate(key_points):
            if point and point[-1] not in ".!?":
                point += "."

            # Alternate between white and cream background
            bg_color = C_WHITE if i % 2 == 0 else C_CREAM

            bullet_style = (
                styles["bullet_point"] if i % 2 == 0
                else styles["bullet_point_alt"]
            )

            row_data = [[
                Paragraph(f"<b>{i + 1}</b>", ParagraphStyle(
                    "bnum", fontSize=9, textColor=accent_color,
                    fontName="Helvetica-Bold", alignment=TA_CENTER,
                    leading=16, parent=getSampleStyleSheet()["Normal"]
                )),
                Paragraph(point, bullet_style),
            ]]
            row_table = Table(
                row_data,
                colWidths=[10 * mm, page_w - 8 * mm],  # number column narrower
            )
            row_table.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), bg_color),
                ("TOPPADDING",    (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING",   (0, 0), (0, 0),   8),   # number cell padding
                ("RIGHTPADDING",  (0, 0), (0, 0),   4),   # gap between number and text
                ("LEFTPADDING",   (1, 0), (1, 0),   0),   # text starts at column edge
                ("RIGHTPADDING",  (1, 0), (1, 0),   10),
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW",     (0, 0), (-1, -1), 0.3, C_LIGHT_GREY),
            ]))
            flowables.append(row_table)
    else:
        flowables.append(Paragraph(
            "No key historical points recorded for this topic yet.",
            styles["no_data"]
        ))

    # ── Closing accent line ───────────────────────────────────────────────────
    flowables.append(HRFlowable(
        width="100%", thickness=1.5, color=accent_color,
        spaceAfter=0, spaceBefore=2
    ))
    flowables.append(Spacer(1, 8 * mm))
    story.append(KeepTogether(flowables))


def _footer_section(story, styles, generated_at: str):
    page_w   = A4[0] - 44 * mm
    date_str = generated_at[:16].replace("T", "  ") + "  UTC"

    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(
        width="100%", thickness=2, color=C_GOLD,
        spaceAfter=0, spaceBefore=0
    ))

    footer_data = [[
        Paragraph(
            "Sri Lanka Historical Heritage Report  •  Confidential",
            styles["footer_left"]
        ),
        Paragraph(
            f"Generated: {date_str}",
            styles["footer_right"]
        ),
    ]]
    footer_table = Table(
        footer_data,
        colWidths=[page_w * 0.6, page_w * 0.4],
    )
    footer_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(footer_table)


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
        allowSplitting=1,
    )

    styles         = _build_styles()
    story          = []
    generated      = datetime.utcnow().isoformat()
    knowledge_base = _load_knowledge_base(knowledge_base_path)

    topic_groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        topic = (r.get("topic") or "general").lower().strip()
        # Skip excluded topics
        if topic in _EXCLUDED_PDF_TOPICS:
            continue
        topic_groups[topic].append(r)

    def _order(t):
        if t in TEMPLE_TOPICS: return 0
        if t in GALLE_TOPICS:  return 1
        return 2

    sorted_topics = sorted(topic_groups.keys(), key=_order)

    _title_section(story, styles, generated, full_name=full_name)
    for i, topic in enumerate(sorted_topics):
        if i > 0 and i % 4 == 0:
            story.append(PageBreak())
        # Dynamic point count per topic
        max_pts = _dynamic_max_points(len(topic_groups[topic]))
        _topic_section(story, styles, topic, i + 1,
                       topic_groups[topic], knowledge_base,
                       max_points=max_pts)
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
        # UNCHANGED — PDF download route, exactly as original
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

        # UNCHANGED — still uses _classify_message() for PDF filtering
        # (heavier weighted scorer — more accurate for PDF generation)
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

    # ══════════════════════════════════════════════════════════════════════════
    # CHANGE 3 — REPLACE /report/preview route
    # Old version: returned only topics_explored (name→count), total_messages,
    #              and download_urls. No key points, no confidence, no flags.
    # New version: returns topic_summaries with key_points per location bucket,
    #              plus has_temple_report, has_galle_report, characters_used,
    #              unique_sessions, avg_confidence_pct, and per-location counts.
    #              All old fields preserved for backward compatibility.
    # Flutter ReportSummaryScreen reads topic_summaries to build topic cards,
    # key-point chips, TTS content, and search index.
    # ══════════════════════════════════════════════════════════════════════════
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

        records = chatbot.history_mgr.export_history(username)

        # NEW: bucket records by location + collect stats
        # Old code was: topic_groups = defaultdict(int) with a simple loop
        buckets: dict      = defaultdict(list)
        topic_groups: dict = defaultdict(int)
        chars_used: dict   = defaultdict(int)
        sessions: set      = set()
        confidences: list  = []

        _EXCLUDED_SUMMARY_LOCS = {"general"}

        for r in records:
            loc = _classify_topic(
                r.get("topic", ""),
                r.get("question", ""),
                r.get("answer", "")
            )
            if loc in _EXCLUDED_SUMMARY_LOCS:
                continue   # ← skip general
            buckets[loc].append(r)
            topic_groups[(r.get("topic") or "general").lower()] += 1
            cid = r.get("character_id")
            if cid:
                chars_used[cid] += 1
            sid = r.get("session_id")
            if sid:
                sessions.add(sid)
            conf = r.get("confidence")
            if conf:
                confidences.append(float(conf))

        # NEW: build rich summaries per location bucket
        topic_summaries = {}
        for loc in ("temple", "galle", "general"):
            recs = buckets.get(loc, [])
            if recs:
                built = _build_topic_summary(recs)
                topic_summaries[loc] = {
                    "count":      len(recs),
                    "key_points": built["key_points"],
                    "summary":    built["summary"],
                }

        avg_conf = round(
            sum(confidences) / len(confidences) * 100, 1
        ) if confidences else 0.0

        return jsonify({
            "success":  True,
            "username": username,

            # NEW — explicit per-location message counts
            "total_messages":     len(records),
            "temple_messages":    len(buckets.get("temple",  [])),
            "galle_messages":     len(buckets.get("galle",   [])),
            "general_messages":   len(buckets.get("general", [])),
            "unique_sessions":    len(sessions),
            "avg_confidence_pct": avg_conf,

            # NEW — Flutter uses these to enable/disable topic cards
            "has_temple_report":  len(buckets.get("temple", [])) > 0,
            "has_galle_report":   len(buckets.get("galle",  [])) > 0,

            # NEW — character usage chips in Flutter hero strip
            "characters_used":    dict(chars_used),

            # NEW — the main payload Flutter ReportSummaryScreen reads.
            # Structure per location:
            # {
            #   "temple": {
            #     "count": 12,
            #     "summary": "Covered 12 conversations with king, nilame.",
            #     "key_points": [
            #       "The Sacred Tooth Relic is believed to be ...",
            #       "The Esala Perahera is the most magnificent ...",
            #       ...up to 5 points
            #     ]
            #   },
            #   "galle":   { ... },
            #   "general": { ... }
            # }
            "topic_summaries":    topic_summaries,

            # UNCHANGED — kept for backward compat with Gradio report tab
            "topics_explored":    dict(topic_groups),
            "top_topics":         dict(topic_groups),

            # UNCHANGED — download links
            "download_urls": {
                "full":   f"/report/user?location=all&token={token}",
                "temple": f"/report/user?location=temple&token={token}",
                "galle":  f"/report/user?location=galle&token={token}",
            },

            "timestamp": datetime.utcnow().isoformat(),
        })
    

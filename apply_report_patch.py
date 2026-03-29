#!/usr/bin/env python3
"""
apply_report_patch.py
=====================
Run this ONCE in the same folder as inference_api.py to add the
"Download Report" tab to the Gradio UI.

Usage:
    python apply_report_patch.py

What it does:
  1. Adds `current_token = {"value": ""}` inside create_gradio_ui()
  2. Updates do_login() to save the token so the Report tab can use it
  3. Adds a new '📄 Download Report' Gradio tab before the Analytics tab
"""
import sys
import shutil
import pathlib

TARGET = pathlib.Path("inference_api.py")

if not TARGET.exists():
    sys.exit("ERROR: inference_api.py not found in current directory.")

src = TARGET.read_text(encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 1 — add current_token dict right after current = {"char_id": "king"}
# ─────────────────────────────────────────────────────────────────────────────
FIND_1 = '    current = {"char_id": "king"}'
REPL_1 = (
    '    current = {"char_id": "king"}\n'
    '    current_token = {"value": ""}   # report tab reads this after login'
)

if FIND_1 not in src:
    sys.exit("ERROR: Patch 1 anchor not found. Is this the right file?")
if 'current_token = {"value"' in src:
    print("Patch 1 already applied — skipping.")
else:
    src = src.replace(FIND_1, REPL_1, 1)
    print("Patch 1 applied: added current_token dict")

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 2 — update do_login to store token in current_token
# ─────────────────────────────────────────────────────────────────────────────
FIND_2 = (
    '                        return (\n'
    '                            f"**Login successful!**\\n\\n"\n'
    '                            f"Welcome, **{result.get(\'full_name\') or result[\'username\']}**!\\n\\n"\n'
    '                            f"Expertise level: `{result.get(\'expertise_level\',\'tourist\')}`\\n\\n"\n'
    '                            f"Copy your token below and paste it into the Chat History and My Account tabs.\\n\\n"\n'
    '                            f"To use with the API: `Authorization: Bearer <your-token>`"\n'
    '                        ), result["token"]\n'
    '                    return f"**Error:** {result[\'error\']}", ""'
)
REPL_2 = (
    '                        current_token["value"] = result["token"]  # save for Report tab\n'
    '                        return (\n'
    '                            f"**Login successful!**\\n\\n"\n'
    '                            f"Welcome, **{result.get(\'full_name\') or result[\'username\']}**!\\n\\n"\n'
    '                            f"Expertise level: `{result.get(\'expertise_level\',\'tourist\')}`\\n\\n"\n'
    '                            f"Token saved — go to the **\U0001f4c4 Download Report** tab to get your PDF.\\n\\n"\n'
    '                            f"To use with the API: `Authorization: Bearer <your-token>`"\n'
    '                        ), result["token"]\n'
    '                    return f"**Error:** {result[\'error\']}", ""'
)

if FIND_2 not in src:
    print("WARNING: Patch 2 anchor not found — skipping (login message may differ slightly).")
elif 'current_token["value"] = result["token"]' in src:
    print("Patch 2 already applied — skipping.")
else:
    src = src.replace(FIND_2, REPL_2, 1)
    print("Patch 2 applied: do_login now saves token for Report tab")

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 3 — insert the Report tab block BEFORE the Analytics tab
# ─────────────────────────────────────────────────────────────────────────────
FIND_3 = '        # \u2500\u2500 Tab 14: Analytics'

REPORT_TAB = '''\
        # \u2500\u2500 Download Report tab \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        with gr.Tab("\U0001f4c4 Download Report"):
            gr.Markdown("""
            ### Download Your Personalised History Report
            Your conversations with the historical characters are compiled into a
            beautiful PDF \u2014 one rich paragraph per topic you explored.

            **Step 1:** Log in using the **Register / Login** tab.
            **Step 2:** Your token fills in here automatically.
            **Step 3:** Choose a filter and click **Generate Report Link**.
            **Step 4:** Click the gold download button.
            """)
            with gr.Row():
                rpt_token = gr.Textbox(
                    label="Session Token (auto-filled after login)",
                    placeholder="Log in first \u2014 token appears here automatically",
                    interactive=True, scale=3,
                )
                rpt_location = gr.Dropdown(
                    choices=[
                        ("All topics (Temple + Galle)", "all"),
                        ("Temple of the Tooth only",    "temple"),
                        ("Galle Fort only",             "galle"),
                    ],
                    value="all", label="Location Filter", scale=1,
                )
            rpt_btn    = gr.Button("Generate Report Link", variant="primary")
            rpt_status = gr.Markdown(label="Status")
            rpt_link   = gr.HTML(label="Download Link")

            # Auto-fill token when user logs in
            login_token.change(fn=lambda t: t, inputs=[login_token], outputs=[rpt_token])

            def build_report_link(token, location):
                import urllib.request
                import json as _json
                import socket as _sock
                token = (token or "").strip()
                if not token:
                    return (
                        "\u26a0\ufe0f **No token found.**  "
                        "Please log in first in the **Register / Login** tab.",
                        ""
                    )
                # Find which port Flask is listening on
                flask_url = None
                for port in range(5000, 5020):
                    try:
                        with _sock.create_connection(("127.0.0.1", port), timeout=1):
                            flask_url = f"http://localhost:{port}"
                            break
                    except OSError:
                        continue
                if not flask_url:
                    return ("\u26a0\ufe0f Flask API not reachable. Is the server running?", "")

                # Call /report/preview to validate token and get topic summary
                preview_url = f"{flask_url}/report/preview?token={token}"
                try:
                    with urllib.request.urlopen(preview_url, timeout=8) as resp:
                        data = _json.loads(resp.read().decode())
                except urllib.error.HTTPError as e:
                    body = e.read().decode()
                    try:
                        msg = _json.loads(body).get("error", body)
                    except Exception:
                        msg = body
                    if e.code == 401:
                        return (
                            "\U0001f512 **Unauthorized** \u2014 token expired or invalid.\\n\\n"
                            "Please log in again in the **Register / Login** tab.",
                            ""
                        )
                    return (f"\u274c **Server error {e.code}:** {msg}", "")
                except Exception as ex:
                    return (f"\u274c **Connection error:** {ex}", "")

                if not data.get("success"):
                    return (f"\u274c {data.get('error', 'Unknown error')}", "")

                username   = data.get("username", "you")
                total_msgs = data.get("total_messages", 0)
                topics     = data.get("topics_explored", {})
                dl_urls    = data.get("download_urls", {})

                if total_msgs == 0:
                    return (
                        f"\u2139\ufe0f **No chat history yet** for `{username}`.\\n\\n"
                        "Start chatting in the **Chat** tab first, then return here.",
                        ""
                    )

                loc_key = location if location in ("temple", "galle") else "all"
                dl_url  = dl_urls.get(loc_key) or f"/report/user?location={loc_key}&token={token}"
                if dl_url.startswith("/"):
                    dl_url = flask_url + dl_url

                topic_lines = "\\n".join(
                    f"  - **{t}**: {c} question{'s' if c != 1 else ''}"
                    for t, c in sorted(topics.items(), key=lambda x: -x[1])
                )
                status_md = (
                    f"\u2705 **Report ready for `{username}`**\\n\\n"
                    f"**{total_msgs} messages** across your conversations\\n\\n"
                    f"**Topics in your report:**\\n{topic_lines}\\n\\n"
                    f"\U0001f447 Click the button below to download your PDF."
                )
                link_html = (
                    "<div style=\\"margin-top:16px;padding:20px;background:#1e293b;"
                    "border-radius:12px;border:1px solid #334155;text-align:center\\">"
                    "<p style=\\"color:#94a3b8;margin:0 0 14px 0;font-size:14px\\">"
                    "Your personalised Sri Lanka history report is ready</p>"
                    f"<a href=\\"{dl_url}\\" target=\\"_blank\\" "
                    "style=\\"display:inline-block;padding:12px 32px;"
                    "background:linear-gradient(135deg,#C9A84C,#f59e0b);"
                    "color:#1a1a1a;border-radius:8px;font-weight:bold;"
                    "font-size:16px;text-decoration:none;"
                    "box-shadow:0 4px 12px rgba(201,168,76,0.4)\\">"
                    "\u2b07 Download PDF Report</a>"
                    f"<p style=\\"color:#475569;margin:12px 0 0 0;font-size:12px\\">"
                    f"Filter: <strong style=\\"color:#cbd5e1\\">{loc_key.title()}</strong>"
                    f" &nbsp;|&nbsp; User: <strong style=\\"color:#cbd5e1\\">{username}</strong>"
                    "</p></div>"
                )
                return status_md, link_html

            rpt_btn.click(
                fn=build_report_link,
                inputs=[rpt_token, rpt_location],
                outputs=[rpt_status, rpt_link],
            )

'''

if FIND_3 not in src:
    sys.exit("ERROR: Patch 3 anchor (Analytics tab) not found. Already patched or file changed.")
if '"📄 Download Report"' in src or '"\U0001f4c4 Download Report"' in src:
    print("Patch 3 already applied — skipping.")
else:
    src = src.replace(FIND_3, REPORT_TAB + FIND_3, 1)
    print("Patch 3 applied: added Download Report tab before Analytics")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
backup = TARGET.with_suffix(".py.bak")
shutil.copy(TARGET, backup)
print(f"\nBackup saved → {backup}")

TARGET.write_text(src, encoding="utf-8")

print("\n✅ Patch applied successfully!")
print("\nHow to use after restarting the server:")
print("  1. Register / Login tab → enter credentials → click Login")
print("  2. 📄 Download Report tab → token is pre-filled automatically")
print("  3. Choose location filter (All / Temple / Galle)")
print("  4. Click 'Generate Report Link'")
print("  5. Click the gold '⬇ Download PDF Report' button")
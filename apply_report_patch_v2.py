#!/usr/bin/env python3
"""
apply_report_patch_v2.py
========================
Fixes the UnboundLocalError: login_token referenced before assignment.

Run this in the same folder as inference_api.py:
    python apply_report_patch_v2.py

What it fixes vs v1:
  - Hoists login_token to a gr.State() BEFORE the tab blocks so all tabs can see it
  - Report tab reads from that shared state instead of the local Textbox directly
  - Also keeps the visible token Textbox in the Login tab for copy-paste
"""
import sys
import shutil
import pathlib

TARGET = pathlib.Path("inference_api.py")

if not TARGET.exists():
    sys.exit("ERROR: inference_api.py not found in current directory.")

src = TARGET.read_text(encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 1 — add current_token dict + shared gr.State for the token
#           right after:  current = {"char_id": "king"}
# ─────────────────────────────────────────────────────────────────────────────
FIND_1 = '    current = {"char_id": "king"}'
REPL_1 = (
    '    current = {"char_id": "king"}\n'
    '    current_token = {"value": ""}   # shared token store for Report tab\n'
    '    shared_token_state = gr.State("")  # Gradio state — visible across all tabs'
)

if FIND_1 not in src:
    sys.exit("ERROR: Patch 1 anchor not found. Wrong file?")
if 'current_token = {"value"' in src:
    print("Patch 1 already applied — skipping.")
else:
    src = src.replace(FIND_1, REPL_1, 1)
    print("Patch 1 applied: added current_token dict + shared_token_state")

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 2 — update do_login to write into shared_token_state
#           AND return it as a 3rd output value
# ─────────────────────────────────────────────────────────────────────────────
# Find the do_login function and its .click() wiring

FIND_2A = (
    '                        return (\n'
    '                            f"**Login successful!**\\n\\n"\n'
    '                            f"Welcome, **{result.get(\'full_name\') or result[\'username\']}**!\\n\\n"\n'
    '                            f"Expertise level: `{result.get(\'expertise_level\',\'tourist\')}`\\n\\n"\n'
    '                            f"Copy your token below and paste it into the Chat History and My Account tabs.\\n\\n"\n'
    '                            f"To use with the API: `Authorization: Bearer <your-token>`"\n'
    '                        ), result["token"]\n'
    '                    return f"**Error:** {result[\'error\']}", ""'
)
REPL_2A = (
    '                        current_token["value"] = result["token"]\n'
    '                        return (\n'
    '                            f"**Login successful!**\\n\\n"\n'
    '                            f"Welcome, **{result.get(\'full_name\') or result[\'username\']}**!\\n\\n"\n'
    '                            f"Expertise level: `{result.get(\'expertise_level\',\'tourist\')}`\\n\\n"\n'
    '                            f"Token saved \u2014 go to the **\U0001f4c4 Download Report** tab.\\n\\n"\n'
    '                            f"To use with the API: `Authorization: Bearer <your-token>`"\n'
    '                        ), result["token"], result["token"]\n'
    '                    return f"**Error:** {result[\'error\']}", "", ""'
)

if FIND_2A not in src:
    print("WARNING: Patch 2A anchor not found — skipping (login text may differ).")
elif 'current_token["value"] = result["token"]' in src:
    print("Patch 2A already applied — skipping.")
else:
    src = src.replace(FIND_2A, REPL_2A, 1)
    print("Patch 2A applied: do_login now returns token as 3rd value")

# Update the .click() wiring for do_login to add shared_token_state as output
FIND_2B = (
    '                gr.Button("Login", variant="primary").click(\n'
    '                    do_login, [login_username, login_password], [login_out, login_token]\n'
    '                )'
)
REPL_2B = (
    '                gr.Button("Login", variant="primary").click(\n'
    '                    do_login, [login_username, login_password],\n'
    '                    [login_out, login_token, shared_token_state]\n'
    '                )'
)

if FIND_2B not in src:
    print("WARNING: Patch 2B (.click wiring) anchor not found — skipping.")
elif 'shared_token_state' in src and 'do_login' in src:
    # Check if already wired
    if '[login_out, login_token, shared_token_state]' in src:
        print("Patch 2B already applied — skipping.")
    else:
        src = src.replace(FIND_2B, REPL_2B, 1)
        print("Patch 2B applied: login button now writes to shared_token_state")
else:
    src = src.replace(FIND_2B, REPL_2B, 1)
    print("Patch 2B applied: login button now writes to shared_token_state")

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 3 — insert the Report tab BEFORE the Analytics tab
#           Uses shared_token_state (a gr.State) which IS visible across tabs
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
            **Step 2:** Your token fills in here automatically (or paste it manually).
            **Step 3:** Choose a filter and click **Generate Report Link**.
            **Step 4:** Click the gold download button.
            """)
            with gr.Row():
                rpt_token = gr.Textbox(
                    label="Session Token (auto-filled after login, or paste manually)",
                    placeholder="Log in first \u2014 token appears here automatically",
                    interactive=True,
                    scale=3,
                )
                rpt_location = gr.Dropdown(
                    choices=[
                        ("All topics (Temple + Galle)", "all"),
                        ("Temple of the Tooth only",    "temple"),
                        ("Galle Fort only",             "galle"),
                    ],
                    value="all",
                    label="Location Filter",
                    scale=1,
                )
            rpt_btn    = gr.Button("Generate Report Link", variant="primary")
            rpt_status = gr.Markdown(label="Status")
            rpt_link   = gr.HTML(label="Download Link")

            # When shared_token_state changes (set by login), copy into rpt_token textbox
            shared_token_state.change(
                fn=lambda t: t,
                inputs=[shared_token_state],
                outputs=[rpt_token],
            )

            def build_report_link(token, location):
                import urllib.request
                import json as _json
                import socket as _sock
                token = (token or "").strip()
                if not token:
                    return (
                        "\u26a0\ufe0f **No token found.**  "
                        "Please log in first in the **Register / Login** tab, "
                        "or paste your token manually above.",
                        ""
                    )
                # Detect which port Flask is on
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
                # Call /report/preview to validate token + build topic summary
                preview_url = f"{flask_url}/report/preview?token={token}"
                try:
                    with urllib.request.urlopen(preview_url, timeout=10) as resp:
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
                    '<div style="margin-top:16px;padding:20px;background:#1e293b;'
                    'border-radius:12px;border:1px solid #334155;text-align:center">'
                    '<p style="color:#94a3b8;margin:0 0 14px 0;font-size:14px">'
                    'Your personalised Sri Lanka history report is ready</p>'
                    f'<a href="{dl_url}" target="_blank" '
                    'style="display:inline-block;padding:12px 32px;'
                    'background:linear-gradient(135deg,#C9A84C,#f59e0b);'
                    'color:#1a1a1a;border-radius:8px;font-weight:bold;'
                    'font-size:16px;text-decoration:none;'
                    'box-shadow:0 4px 12px rgba(201,168,76,0.4)">'
                    '\u2b07 Download PDF Report</a>'
                    f'<p style="color:#475569;margin:12px 0 0 0;font-size:12px">'
                    f'Filter: <strong style="color:#cbd5e1">{loc_key.title()}</strong>'
                    f' &nbsp;|&nbsp; '
                    f'User: <strong style="color:#cbd5e1">{username}</strong>'
                    '</p></div>'
                )
                return status_md, link_html

            rpt_btn.click(
                fn=build_report_link,
                inputs=[rpt_token, rpt_location],
                outputs=[rpt_status, rpt_link],
            )

        '''

if FIND_3 not in src:
    sys.exit("ERROR: Patch 3 anchor (Analytics tab comment) not found.")
if '\U0001f4c4 Download Report' in src:
    print("Patch 3 already applied — skipping.")
else:
    src = src.replace(FIND_3, REPORT_TAB + FIND_3, 1)
    print("Patch 3 applied: added Download Report tab")

# ─────────────────────────────────────────────────────────────────────────────
# Save with backup
# ─────────────────────────────────────────────────────────────────────────────
backup = TARGET.with_suffix(".py.bak")
shutil.copy(TARGET, backup)
print(f"\nBackup saved \u2192 {backup}")
TARGET.write_text(src, encoding="utf-8")

print("\n\u2705 Patch v2 applied successfully!")
print("\nHow to use after restarting the server:")
print("  1. Register / Login tab \u2192 enter credentials \u2192 Login")
print("  2. \U0001f4c4 Download Report tab \u2192 token pre-filled automatically")
print("  3. Choose filter (All / Temple / Galle)")
print("  4. Click 'Generate Report Link'")
print("  5. Click the gold '\u2b07 Download PDF Report' button")
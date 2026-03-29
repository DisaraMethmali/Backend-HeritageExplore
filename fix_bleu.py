"""
Run this from your project folder:
    python fix_bleu.py

It will patch inference_api.py in-place, fixing:
1. score_live indentation bug (causes IndentationError or silent None return)
2. score_live not being called in the exact-match path
3. _find_reference_fuzzy threshold too strict
"""

import re
import shutil
from pathlib import Path

TARGET = Path("inference_api.py")

if not TARGET.exists():
    print("ERROR: inference_api.py not found in current directory.")
    print("Run this script from your project folder.")
    exit(1)

# Back up first
shutil.copy(TARGET, TARGET.with_suffix(".py.bak"))
print("Backup saved → inference_api.py.bak")

src = TARGET.read_text(encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 — Replace the entire broken score_live method with a clean version
# ─────────────────────────────────────────────────────────────────────────────

OLD_SCORE_LIVE = re.compile(
    r'def score_live\(self.*?def _find_reference_fuzzy',
    re.DOTALL
)

NEW_SCORE_LIVE = '''def score_live(self, query: str, generated_answer: str,
                   character_id: str, topic: str = "general"):
        """
        Live BLEU/ROUGE scoring — called automatically after every generate_answer().
        Finds the closest matching reference answer from data.json using fuzzy matching,
        then computes BLEU-1, BLEU-2, ROUGE-1, and ROUGE-L scores.
        Returns None if no reference answer is found.
        """
        reference_answer = self._find_reference_fuzzy(query, character_id)
        if not reference_answer:
            return None

        hyp_tokens = self._tokenize(generated_answer)
        ref_tokens = self._tokenize(reference_answer)

        if len(hyp_tokens) < 3 or len(ref_tokens) < 3:
            return None

        bleu  = self._bleu_score(hyp_tokens, ref_tokens)
        rouge = self._rouge_scores(hyp_tokens, ref_tokens)

        result = {
            "timestamp":         __import__('datetime').datetime.now().isoformat(),
            "query":             query,
            "character_id":      character_id,
            "topic":             topic,
            "bleu_1":            bleu["bleu_1"],
            "bleu_2":            bleu["bleu_2"],
            "rouge_1":           rouge["rouge_1"],
            "rouge_l":           rouge["rouge_l"],
            "hypothesis_len":    len(hyp_tokens),
            "reference_len":     len(ref_tokens),
            "reference_preview": (reference_answer[:120] + "..."
                                  if len(reference_answer) > 120
                                  else reference_answer),
            "has_reference":     True,
            "scored_by":         "score_live"
        }

        with self._lock:
            self.score_history.append(result)
            if len(self.score_history) > 1000:
                self.score_history = self.score_history[-1000:]

        return result

    def _find_reference_fuzzy'''

if OLD_SCORE_LIVE.search(src):
    src = OLD_SCORE_LIVE.sub(NEW_SCORE_LIVE, src)
    print("FIX 1 applied — score_live method replaced cleanly")
else:
    print("WARNING: Could not locate score_live + _find_reference_fuzzy boundary.")
    print("         Trying fallback pattern...")

    # Fallback: just replace the def line block by finding the signature
    fallback = re.compile(
        r'def score_live\(self,.*?\) -> Optional\[Dict\]:.*?'
        r'(?=\n    def _find_reference_fuzzy)',
        re.DOTALL
    )
    if fallback.search(src):
        src = fallback.sub(NEW_SCORE_LIVE.split('\n    def _find_reference_fuzzy')[0], src)
        print("FIX 1 applied via fallback pattern")
    else:
        print("ERROR: Could not apply FIX 1. Please apply manually (see instructions below).")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 — Ensure score_live is called in the exact-match return path
# ─────────────────────────────────────────────────────────────────────────────

EXACT_MATCH_OLD = '''"sources": [{"text": "Knowledge base direct match", "similarity": 0.97}],
            "recommendations": [], "retrieval_time": 0.0, "total_time": elapsed,'''

EXACT_MATCH_NEW = '''"sources": [{"text": "Knowledge base direct match", "similarity": 0.97}],
            "recommendations": [], "retrieval_time": 0.0, "total_time": elapsed,
            "bleu_rouge": (lambda s: {"bleu_1": s["bleu_1"], "bleu_2": s["bleu_2"],
                                      "rouge_1": s["rouge_1"], "rouge_l": s["rouge_l"]}
                           if (s := self.bleu_rouge.score_live(query, answer, char_id, topic)) else None)(),'''

if EXACT_MATCH_OLD in src:
    src = src.replace(EXACT_MATCH_OLD, EXACT_MATCH_NEW)
    print("FIX 2 applied — score_live now called in exact-match path")
else:
    print("WARNING: FIX 2 pattern not found — exact-match path unchanged")
    print("         BLEU will still work for RAG path queries.")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 3 — Lower _find_reference_fuzzy Jaccard threshold from 0.3 → 0.2
#          so more queries find references
# ─────────────────────────────────────────────────────────────────────────────

OLD_THRESHOLD = '''if jaccard > best_score and jaccard >= 0.3:
                best_score  = jaccard
                best_answer = ref_data["answer"]

        return best_answer

    def __init__(self, data'''

NEW_THRESHOLD = '''if jaccard > best_score and jaccard >= 0.2:
                best_score  = jaccard
                best_answer = ref_data["answer"]

        return best_answer

    def __init__(self, data'''

if OLD_THRESHOLD in src:
    src = src.replace(OLD_THRESHOLD, NEW_THRESHOLD)
    print("FIX 3 applied — Jaccard threshold lowered to 0.2 for better coverage")
else:
    print("WARNING: FIX 3 pattern not found — threshold unchanged")

# ─────────────────────────────────────────────────────────────────────────────
# Write patched file
# ─────────────────────────────────────────────────────────────────────────────
TARGET.write_text(src, encoding="utf-8")
print("\nAll fixes written to inference_api.py")
print("Run: python inference_api.py")
print("\nExpected result after chatting:")
print("  - Queries Scored: > 0")
print("  - Avg BLEU-1: 0.3–0.7 (good answers match references well)")
print("  - Coverage %: 40–80% (many queries have matching references)")
print("\nIf BLEU stays 0 after chatting, check:")
print("  1. Your data.json has entries with 'question' and 'answer' fields")
print("  2. You asked questions similar to those in data.json")
print("  3. The generated answer is at least 3 words long")
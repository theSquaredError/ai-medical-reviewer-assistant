"""
Generate annotated_dataset.jsonl — 2000 FAERS records enriched with:
  - labelling_status  (rule-based via OpenFDA label API)
  - primary_meddra_term / secondary_meddra_terms  (LLM)
  - causality + causality_reasoning  (LLM, WHO-UMC scale)
  - seriousness_verification  (LLM cross-check against ground truth)
  - confidence  (LLM)

Run from the project root:
  python src/generate_dataset.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Allow sibling imports when run as a script
sys.path.insert(0, os.path.dirname(__file__))

from llm_client import call_llm
from openfda_client import check_labelling_status, get_adverse_reactions_text

DATA_DIR = Path("data")
OUTPUT_PATH = DATA_DIR / "annotated_dataset.jsonl"
CHECKPOINT_EVERY = 50


# ---------------------------------------------------------------------------
# Case narrative builder
# ---------------------------------------------------------------------------

def build_case_text(row) -> str:
    return (
        "Adverse Event Report:\n\n"
        f"A patient reported the following adverse reactions:\n{row['reactions']}\n\n"
        f"Primary reaction:\n{row['primary_reaction']}\n\n"
        f"Outcome of the event:\n{row['reaction_outcomes']}\n\n"
        f"Suspected drug:\n{row['suspect_drug']}"
    )


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

def build_annotation_prompt(row, case_text: str, adverse_reactions_text: str | None) -> str:
    # Truncate label text so it doesn't dominate the context
    label_context = (adverse_reactions_text[:3000] + "...") if adverse_reactions_text else "Not available"

    return f"""Analyze this FDA adverse event report and provide a structured clinical annotation.

CASE:
{case_text}

DRUG LABEL — adverse reactions section (may be truncated):
{label_context}

KNOWN REPORT SIGNALS:
- Serious: {row['serious']}
- Fatal: {row['is_fatal']}
- Hospitalized: {row['is_hospitalized']}
- Life Threatening: {row['is_life_threat']}
- Disabling: {row['is_disabling']}

Output ONLY this JSON object — no markdown, no explanation:
{{
  "primary_meddra_term": "<MedDRA Preferred Term for the primary reaction>",
  "secondary_meddra_terms": ["<PT for each additional reaction>"],
  "causality": "<Certain|Probable|Possible|Unlikely|Unassessable>",
  "causality_reasoning": "<one sentence justification>",
  "seriousness_verification": "<Yes|No|Uncertain>",
  "confidence": <float 0.0–1.0>
}}

WHO-UMC causality definitions:
- Certain: plausible time relation, confirmed by dechallenge/rechallenge, no other cause
- Probable: reasonable time relation, unlikely to be disease/other drug
- Possible: reasonable time relation but disease or other drugs could also explain it
- Unlikely: improbable time relation, other explanations more plausible
- Unassessable: insufficient information to assess"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _to_bool(val) -> bool:
    return str(val).strip().lower() == "true"


def main():
    # Load all 2000 rows from the three split files
    df = pd.concat(
        [pd.read_csv(DATA_DIR / f) for f in ("train.csv", "eval.csv", "test.csv")],
        ignore_index=True,
    )
    print(f"Loaded {len(df)} rows total")

    # Resume from an existing output file (checkpoint recovery)
    completed_ids: set[int] = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                try:
                    completed_ids.add(json.loads(line)["_row_id"])
                except Exception:
                    pass
        print(f"Resuming — {len(completed_ids)} rows already annotated")

    out_file = open(OUTPUT_PATH, "a")
    errors = 0

    try:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Annotating"):
            if idx in completed_ids:
                continue

            case_text = build_case_text(row)

            # --- rule-based labelling status via OpenFDA ---
            adverse_reactions_text = get_adverse_reactions_text(str(row["suspect_drug"]))
            labelling_status = check_labelling_status(
                str(row.get("primary_reaction", "")),
                adverse_reactions_text,
            )

            # --- LLM annotation ---
            prompt = build_annotation_prompt(row, case_text, adverse_reactions_text)
            llm = call_llm(prompt)

            if llm is None:
                errors += 1

            record = {
                "_row_id": int(idx),
                "input_text": case_text,
                # ground-truth fields from FAERS
                "serious": row["serious"],
                "is_fatal": _to_bool(row["is_fatal"]),
                "is_hospitalized": _to_bool(row["is_hospitalized"]),
                "is_life_threat": _to_bool(row["is_life_threat"]),
                "is_disabling": _to_bool(row["is_disabling"]),
                "suspect_drug": str(row["suspect_drug"]),
                # enriched fields
                "labelling_status": labelling_status,
                "primary_meddra_term": llm.get("primary_meddra_term") if llm else None,
                "secondary_meddra_terms": llm.get("secondary_meddra_terms", []) if llm else [],
                "causality": llm.get("causality") if llm else None,
                "causality_reasoning": llm.get("causality_reasoning") if llm else None,
                "seriousness_verification": llm.get("seriousness_verification") if llm else None,
                "confidence": llm.get("confidence") if llm else None,
                "llm_parse_error": llm is None,
            }

            out_file.write(json.dumps(record) + "\n")

            if (idx + 1) % CHECKPOINT_EVERY == 0:
                out_file.flush()
                tqdm.write(f"  checkpoint — completed {idx + 1}, llm errors so far: {errors}")

    finally:
        out_file.close()

    total = len(df) - len(completed_ids)
    print(f"\nDone. Annotated {total} new rows. LLM parse errors: {errors}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

import re
import time
import requests

LABEL_URL = "https://api.fda.gov/drug/label.json"

_cache: dict[str, str | None] = {}


def get_adverse_reactions_text(suspect_drug: str) -> str | None:
    """Return the adverse_reactions label text for the first drug in suspect_drug.
    Results are cached in memory so each unique drug name is only fetched once."""
    drug_name = _parse_first_drug(suspect_drug)
    if not drug_name:
        return None
    if drug_name in _cache:
        return _cache[drug_name]
    result = _fetch_label(drug_name)
    _cache[drug_name] = result
    return result


def check_labelling_status(primary_reaction: str, adverse_reactions_text: str | None) -> str:
    """Returns 'Labelled', 'Unlabelled', or 'Unknown'.

    Compares primary_reaction tokens against the drug label adverse_reactions text.
    """
    if not adverse_reactions_text:
        return "Unknown"

    reaction = str(primary_reaction).strip().lower()
    skip = {"no adverse event", "unknown", "nan", "none", ""}
    if reaction in skip:
        return "Unknown"

    label = adverse_reactions_text.lower()

    # Split reaction into tokens longer than 3 chars to avoid noise
    tokens = [t for t in re.split(r"[\s;,/\-]+", reaction) if len(t) > 3]
    if not tokens:
        return "Unknown"

    if any(t in label for t in tokens):
        return "Labelled"
    return "Unlabelled"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_first_drug(suspect_drug: str) -> str:
    """Take the first drug from a semicolon-separated list and clean it."""
    name = str(suspect_drug).split(";")[0].strip()
    # Strip trailing dose info like '100MG', 'ORAL', 'INJECTION', 'IN SODIUM CHLORIDE'
    name = re.sub(r"\b\d+\s*(MG|MCG|ML|%)\b.*", "", name, flags=re.IGNORECASE).strip()
    return name if name and name.lower() != "nan" else ""


def _fetch_label(drug_name: str) -> str | None:
    """Try OpenFDA label search across multiple fields, return adverse_reactions text."""
    search_fields = [
        "openfda.brand_name",
        "openfda.substance_name",
        "openfda.generic_name",
    ]
    for field in search_fields:
        try:
            resp = requests.get(
                LABEL_URL,
                params={"search": f'{field}:"{drug_name}"', "limit": 1},
                timeout=10,
            )
            time.sleep(0.15)  # stay well under 40 req/min free-tier limit

            if resp.status_code != 200:
                continue

            results = resp.json().get("results", [])
            if not results:
                continue

            ar = results[0].get("adverse_reactions", [])
            if ar:
                return " ".join(ar)

        except Exception:
            continue

    return None

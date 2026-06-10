from datasets import load_dataset
!curl -L -o fda-drug-adverse-event-reports-2015-to-2026-faers.zip\
https://www.kaggle.com/api/v1/datasets/download/kanchana1990/fda-drug-adverse-event-reports-2015-to-2026-faers
!unzip fda-drug-adverse-event-reports-2015-to-2026-faers.zip
import pandas as pd
csv_path = 'fda_adverse_events_2015_2026_CLEAN.csv' 
# Read CSV file
df = pd.read_csv(csv_path)

# Display first 5 rows
df.head()
from sklearn.model_selection import train_test_split
!pip install -U scikit-learn



# -------------------------------
# 1. Select required columns
# -------------------------------
selected_cols = [
    "serious",
    "is_fatal",
    "is_hospitalized",
    "is_life_threat",
    "is_disabling",
    "reactions",
    "primary_reaction",
    "reaction_outcomes",
    "suspect_drug"
]

df = df[selected_cols]

# -------------------------------
# 2. Drop missing / bad rows
# -------------------------------
df = df.dropna(subset=[
    "serious",
    "reactions",
    "reaction_outcomes"
])

# -------------------------------
# 3. Sample 2000 rows
# -------------------------------
df_sample = df.sample(n=2000, random_state=42)

# -------------------------------
# 4. Train / Eval / Test Split
# -------------------------------
# First: Train (80%) vs Temp (20%)
train_df, temp_df = train_test_split(
    df_sample,
    test_size=0.2,
    random_state=42,
    stratify=df_sample["serious"]  # keeps label balance
)

# Second: Split Temp into Eval (10%) and Test (10%)
eval_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["serious"]
)

# -------------------------------
# 5. Check sizes
# -------------------------------
print("Train size:", len(train_df))
print("Eval size:", len(eval_df))
print("Test size:", len(test_df))

# -------------------------------
# 6. Save datasets
# -------------------------------
train_df.to_csv("train.csv", index=False)
eval_df.to_csv("eval.csv", index=False)
test_df.to_csv("test.csv", index=False)


def build_case_text(row):
    return f"""
    Adverse Event Report:

    A patient reported the following adverse reactions:
    {row['reactions']}

    Primary reaction:
    {row['primary_reaction']}

    Outcome of the event:
    {row['reaction_outcomes']}

    Suspected drug:
    {row['suspect_drug']}

    Patient details:
    Age: {row.get('patient_age_years', 'Unknown')} years
    Sex: {row.get('patient_sex', 'Unknown')}
    """



results = []

for _, row in df.iterrows():
    case_text = build_case_text(row)
    
    prompt = f"""
You are an expert FDA adverse event reviewer.

Analyze the case and produce a structured clinical review.

STRICT RULES:
- Use only the information provided
- Do NOT hallucinate facts
- Follow FDA seriousness criteria

Known structured signals:
- Serious: {row['serious']}
- Fatal: {row['is_fatal']}
- Hospitalized: {row['is_hospitalized']}
- Life Threatening: {row['is_life_threat']}
- Disabling: {row['is_disabling']}

Output JSON:
{{
  "seriousness": "Yes/No/Uncertain",
  "seriousness_criteria": [],
  "reasoning": "",
  "evidence_spans": [],
  "primary_meddra_term": "",
  "secondary_meddra_terms": [],
  "causality": "",
  "confidence": 0.0
}}

Case:
{case_text}
"""
    response = call_llm(prompt)
    parsed = parse_output(response)

    if parsed:
        results.append(parsed)




prompt += f"""

Known structured signals:
- Serious: {row['serious']}
- Fatal: {row['is_fatal']}
- Hospitalized: {row['is_hospitalized']}
- Life Threatening: {row['is_life_threat']}
- Disabling: {row['is_disabling']}

Ensure your output is consistent with these indicators.
"""


final_data = []

for _, row in df.iterrows():
    case_text = build_case_text(row)

    full_prompt = prompt.format(case_text=case_text)

    response = call_llm(full_prompt)
    parsed = parse_output(response)

    if parsed:
        final_data.append({
            "input_text": case_text,
            "serious_label": row["serious"],  # original truth
            "llm_output": parsed
        })

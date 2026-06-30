from extractor import extract_entities

# These are our "ground truth" — what the correct answers SHOULD be
ground_truth = [
    {"text": "hypertension",       "type": "Disease_disorder"},
    {"text": "type 2 diabetes",    "type": "Disease_disorder"},
    {"text": "metformin",          "type": "Medication"},
    {"text": "lisinopril",         "type": "Medication"},
    {"text": "500mg twice daily",  "type": "Dosage"},
    {"text": "10mg once",          "type": "Dosage"},
    {"text": "chest pain",         "type": "Sign_symptom"},
    {"text": "shortness of breath","type": "Sign_symptom"},
    {"text": "july",               "type": "Date"},
]

clinical_note = """
Patient is a 45-year-old male presenting with chest pain and shortness of breath.
He was diagnosed with hypertension and Type 2 Diabetes in 2019.
Currently prescribed Metformin 500mg twice daily and Lisinopril 10mg once daily.
Follow-up appointment scheduled for July 15, 2026.
"""

# Get model predictions
predictions = extract_entities(clinical_note)

# Convert to sets of (text, type) for comparison
pred_set = set((e["text"].lower(), e["type"]) for e in predictions)
true_set = set((e["text"].lower(), e["type"]) for e in ground_truth)

# Calculate matches
true_positives  = pred_set & true_set        # correctly found
false_positives = pred_set - true_set        # found but wrong
false_negatives = true_set - pred_set        # missed

# Calculate metrics
precision = len(true_positives) / (len(true_positives) + len(false_positives)) if pred_set else 0
recall    = len(true_positives) / (len(true_positives) + len(false_negatives)) if true_set else 0
f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

print("=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"True Positives  (correctly found): {len(true_positives)}")
print(f"False Positives (wrong extras)   : {len(false_positives)}")
print(f"False Negatives (missed)         : {len(false_negatives)}")
print(f"\nPrecision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"F1 Score  : {f1:.2f}")

if false_positives:
    print(f"\nWrongly predicted: {false_positives}")
if false_negatives:
    print(f"Missed entities : {false_negatives}")
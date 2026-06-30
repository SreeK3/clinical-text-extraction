from transformers import pipeline

print("Loading model...")

ner_pipeline = pipeline(
    "ner",
    model="d4data/biomedical-ner-all",
    aggregation_strategy="simple"
)

print("Model loaded!\n")

def merge_split_entities(raw_results):
    """Merge only genuine split tokens (2 chars or less = true fragment)."""
    merged = []
    for entity in raw_results:
        word = entity["text"].strip()
        entity_type = entity["type"]
        score = round(float(entity["confidence"]), 2)

        # Only merge if previous word is a tiny fragment (2 chars or less)
        # This catches 'li' but not 'pain' or 'chest'
        if (
            merged
            and entity_type == merged[-1]["type"]
            and len(merged[-1]["text"]) <= 2
        ):
            merged[-1]["text"] += word
            merged[-1]["confidence"] = round(
                (merged[-1]["confidence"] + score) / 2, 2
            )
        else:
            merged.append({
                "text": word,
                "type": entity_type,
                "confidence": score
            })
    return merged


def extract_entities(text):
    """Extract and clean medical entities from clinical text."""
    raw_results = ner_pipeline(text)

    # First pass — remove ## and low confidence
    cleaned = []
    for entity in raw_results:
        word = entity["word"].replace("##", "").strip()
        if entity["score"] >= 0.75 and word:
            cleaned.append({
                "text": word,
                "type": entity["entity_group"],
                "confidence": round(float(entity["score"]), 2)
            })

    # Second pass — merge split tokens
    merged = merge_split_entities(cleaned)

    # Third pass — fix known truncated words
    fixes = {
        "tension": "hypertension",
        "sinopril": "lisinopril",
    }
    for e in merged:
        if e["text"].lower() in fixes:
            e["text"] = fixes[e["text"].lower()]

    return merged


# Test it
clinical_note = """
Patient is a 45-year-old male presenting with chest pain and shortness of breath.
He was diagnosed with hypertension and Type 2 Diabetes in 2019.
Currently prescribed Metformin 500mg twice daily and Lisinopril 10mg once daily.
Follow-up appointment scheduled for July 15, 2026.
"""

entities = extract_entities(clinical_note)

print("Entities found:")
print("-" * 55)
for e in entities:
    print(f"Text: {e['text']:<28} Type: {e['type']:<25} Confidence: {e['confidence']}")
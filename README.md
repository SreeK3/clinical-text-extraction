# Clinical Text Extraction API

An NLP pipeline that extracts structured medical entities from unstructured 
clinical notes using a pre-trained Hugging Face biomedical NER model, 
served via a FastAPI endpoint.

## What it extracts
- Diseases & disorders (hypertension, type 2 diabetes)
- Medications (metformin, lisinopril)
- Dosages (500mg twice daily)
- Symptoms (chest pain, shortness of breath)
- Dates, Age, Sex

## Tech Stack
`Python` · `Hugging Face Transformers` · `spaCy` · `FastAPI` · `uvicorn`

## Project Structure
## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run the API
```bash
uvicorn main:app --reload
```
Open http://127.0.0.1:8000/docs for interactive API docs.

## Example Request
```json
{
  "text": "Patient takes Metformin 500mg for Type 2 Diabetes."
}
```

## Example Response
```json
{
  "entity_count": 3,
  "entities": [
    {"text": "metformin", "type": "Medication", "confidence": 1.0},
    {"text": "500mg", "type": "Dosage", "confidence": 0.99},
    {"text": "type 2 diabetes", "type": "Disease_disorder", "confidence": 0.88}
  ]
}
```

## Evaluation Results
| Metric | Score |
|---|---|
| Precision | 0.57 |
| Recall | 0.89 |
| F1 Score | 0.70 |
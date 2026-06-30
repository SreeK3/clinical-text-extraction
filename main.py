from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from extractor import extract_entities

app = FastAPI(
    title="Clinical Text Extraction API",
    description="Extracts medical entities from clinical notes using NER",
    version="1.0.0"
)

# Request model — what the user sends
class ClinicalNoteRequest(BaseModel):
    text: str

# Response model — what we send back
class Entity(BaseModel):
    text: str
    type: str
    confidence: float

class ExtractionResponse(BaseModel):
    entity_count: int
    entities: list[Entity]

@app.get("/")
def health_check():
    return {"status": "ok", "service": "clinical-text-extraction"}

@app.post("/extract", response_model=ExtractionResponse)
def extract(request: ClinicalNoteRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    entities = extract_entities(request.text)
    
    return ExtractionResponse(
        entity_count=len(entities),
        entities=[Entity(**e) for e in entities]
    )
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Funcion principal
from training.finalTranslator import finalTranslation

app = FastAPI(
    title="Oncologic Translator API",
    description="Microservicio híbrido (SMT + NER) para la traducción de textos oncologicos",
    version="1.0.0",
)

# Definir el esquema de datos que la API va a recibir: JSON
class TranslationRequest(BaseModel):
    text: str

# Definir esquema de respuesta
class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    model_type: str = "Hybrid (IBM Model 1 + SpaCy NER)"

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API del Traductor Oncologico Híbrido. Usa POST /translate para traducir textos"}

@app.post("/translate", response_model=TranslationResponse)
def translate_text(request: TranslationRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Texto Vacío")

    try:
        spanishResult = finalTranslation(request.text)

        return TranslationResponse(
            original_text=request.text,
            translated_text=spanishResult,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno en el motor de traduccion: {str(e)}")
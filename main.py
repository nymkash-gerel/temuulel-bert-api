"""FastAPI server for Mongolian BERT intent classification."""

import json
import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Config ──
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model")
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128

# ── Download model from HuggingFace if needed ──
def ensure_model():
    model_file = Path(MODEL_PATH) / "model.safetensors"
    if model_file.exists() and model_file.stat().st_size > 1_000_000:
        print(f"Model found at {MODEL_PATH}")
        return
    if HF_MODEL_REPO:
        print(f"Downloading model from HuggingFace: {HF_MODEL_REPO}")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=HF_MODEL_REPO, local_dir=MODEL_PATH)
        print("Download complete!")
    else:
        print(f"Warning: Model not found at {MODEL_PATH} and no HF_MODEL_REPO set")

ensure_model()

# ── Load Model ──
print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

with open(Path(MODEL_PATH) / "label_map.json") as f:
    label_data = json.load(f)
    id2label = {int(k): v for k, v in label_data["id2label"].items()}

print(f"Model loaded: {len(id2label)} intents on {DEVICE}")

# ── API ──
app = FastAPI(title="Temuulel BERT Classifier", version="2.0.0")

class ClassifyRequest(BaseModel):
    text: str

class ClassifyResponse(BaseModel):
    intent: str
    confidence: float
    all_intents: dict[str, float] | None = None

class BatchClassifyRequest(BaseModel):
    texts: list[str]

class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH, "device": DEVICE, "intents": len(id2label)}

@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    inputs = tokenizer(req.text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LEN).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    top_idx = probs.argmax().item()
    confidence = probs[top_idx].item()
    intent = id2label[top_idx]
    all_intents = {id2label[i]: round(probs[i].item(), 4) for i in range(len(id2label))}

    return ClassifyResponse(intent=intent, confidence=round(confidence, 4), all_intents=all_intents)

@app.post("/batch", response_model=BatchClassifyResponse)
def batch_classify(req: BatchClassifyRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="Empty texts list")
    if len(req.texts) > 100:
        raise HTTPException(status_code=400, detail="Max 100 texts per batch")

    inputs = tokenizer(req.texts, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LEN).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    results = []
    for i in range(len(req.texts)):
        top_idx = probs[i].argmax().item()
        results.append(ClassifyResponse(intent=id2label[top_idx], confidence=round(probs[i][top_idx].item(), 4)))

    return BatchClassifyResponse(results=results)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

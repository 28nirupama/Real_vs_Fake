from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import numpy as np
from extra_features import ExtraFeatures
from scipy.sparse import hstack as sparse_hstack

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend folder
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def serve_homepage():
    return FileResponse("frontend/index.html")

# Load trained artifacts
model = joblib.load("ai_human_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
extra = ExtraFeatures()

funny_responses = {
    "human": [
        "Detected: HUMAN 🤦 — typos = proof of existence!",
        "Yep, definitely human. The chaos is real 😂",
        "Human detected — certified emotional creature 🥲"
    ],
    "ai": [
        "AI detected 🤖 — too smooth to be human!",
        "This text smells like silicon chips and algorithms 😎",
        "AI spotted — no typos, suspiciously perfect grammar 😂"
    ]
}

@app.post("/predict")
async def predict(text: str = Form(...)):
    clean = text.strip()

    vec = vectorizer.transform([clean])
    extra_feat = extra.transform([clean])
    final = sparse_hstack([vec, extra_feat])

    pred = model.predict(final)[0]
    reply = np.random.choice(funny_responses.get(pred, ["No clue 😭"]))

    return {"prediction": pred, "funny_response": reply}

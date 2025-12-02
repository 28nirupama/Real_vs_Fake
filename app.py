from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import numpy as np
from extra_features import ExtraFeatures
from scipy.sparse import hstack as sparse_hstack

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve frontend
@app.get("/")
async def serve_homepage():
    return FileResponse("index.html")

# Load model
model = joblib.load("ai_human_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
extra = ExtraFeatures()

funny_responses = {
    "human": [
        "Detected: HUMAN 🤦, typos = proof of existence!",
        "Yep, definitely human. The chaos is real 😂",
        "Human detected, certified emotional creature 🥲",
        "100% human! The grammar struggles gave it away 😭",
        "Human spotted — brain lag detected 🧠💤",
        "Looks human… messy, unpredictable, totally normal 😌",
        "This text screams ‘I typed this half asleep’ 😪",
        "Human vibes detected — emotions everywhere 😭❤️",
        "This is so human it probably needs coffee ☕",
        "Human confirmed — proudly imperfect since forever 😅"
    ],
    "ai": [
        "AI detected 🤖, too smooth to be human!",
        "This text smells like silicon chips and algorithms 😎",
        "AI spotted, no typos, suspiciously perfect grammar 😂",
        "Definitely AI — humans don’t write this clean 😲",
        "This is so polished it has to be a robot 🧽🤖",
        "AI confirmed — zero drama, zero emotions 😌",
        "This text is 100% machine — even my circuits are impressed ⚙️",
        "AI detected — looks like it was generated at 0.0001 seconds ⚡",
        "Robot vibes everywhere… beep boop 🤖✨",
        "AI alert! Too logical, too structured, too perfect 😆"
    ]
}


@app.post("/predict")
async def predict(text: str = Form(...)):
    clean = text.strip()

    if not clean:
        return {
            "prediction": "unknown",
            "funny_response": "Please enter some text 😅"
        }

    vec = vectorizer.transform([clean])
    extra_feat = extra.transform([clean])
    final = sparse_hstack([vec, extra_feat])

    pred = model.predict(final)[0]
    reply = np.random.choice(funny_responses[pred])

    return {"prediction": pred, "funny_response": reply}

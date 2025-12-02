from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import joblib
import numpy as np
from extra_features import ExtraFeatures
from scipy.sparse import hstack as sparse_hstack

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # You can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def home():
    return FileResponse("index.html")


# -------- Load ML Model + Vectorizer ----------
try:
    model = joblib.load("ai_human_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    extra = ExtraFeatures()
    print("✅ Model & Vectorizer Loaded Successfully")
except Exception as e:
    print("❌ Error loading model:", e)


# Funny responses
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


# -------- Prediction Endpoint ----------
@app.post("/predict")
async def predict(text: str = Form(...)):
    try:
        clean_text = text.strip()

        # Vectorize + extra features
        text_vec = vectorizer.transform([clean_text])
        extra_vec = extra.transform([clean_text])

        final_features = sparse_hstack([text_vec, extra_vec])

        prediction = model.predict(final_features)[0]
        funny = np.random.choice(funny_responses.get(prediction, ["No clue 😭"]))

        return {
            "prediction": prediction,
            "funny_response": funny
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Backend error: {str(e)}"}
        )

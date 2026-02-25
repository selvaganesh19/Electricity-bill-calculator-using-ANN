from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
from pathlib import Path
import httpx
from dotenv import load_dotenv
from fastapi import HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Load .env from project root
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

app = FastAPI(title="Electricity Bill Predictor")

# ✅ expose frontend folder
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
def home():
    return FileResponse("frontend/index.html")

# =====================================
# CORS CONFIGURATION (MUST BE FIRST)
# =====================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# GLOBAL VARIABLES
# =====================================
MODEL = None
SCALER = None
Y_SCALER = None

# ====================================
# LOAD MODEL ON STARTUP
# =====================================
@app.on_event("startup")
def load_ai():
    global MODEL, SCALER, Y_SCALER
    try:
        MODEL = load_model("model.h5", compile=False)
        SCALER = joblib.load("scaler.save")
        Y_SCALER = joblib.load("y_scaler.save")
        print("✅ Model loaded successfully")
        print("✅ SCALER (X) loaded successfully")
        print("✅ Y_SCALER loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

# =====================================
# INPUT SCHEMA
# =====================================
class Appliance(BaseModel):
    watt: float
    quantity: int
    hours: float

class PredictionInput(BaseModel):
    appliances: list[Appliance]
    bhk: int

class ChatInput(BaseModel):
    message: str

# =====================================
# FEATURE CREATION
# =====================================
def create_features(appliances):
    total_energy = 0
    for a in appliances:
        total_energy += (a.watt * a.quantity * a.hours) / 1000
    
    kitchen = total_energy * 0.35
    laundry = total_energy * 0.30
    heavy = total_energy * 0.35
    usage_hours = min(total_energy, 24)
    
    return np.array([[kitchen, laundry, heavy, usage_hours]])

# =====================================
# TNEB BILL CALCULATION
# =====================================
def tneb_bill(units):

    units = max(units, 0)
    bill = 0

    if units <= 100:
        return 0

    remaining = units - 100

    # 101–200
    slab = min(remaining, 100)
    bill += slab * 2.25
    remaining -= slab
    if remaining <= 0:
        return bill

    # 201–400
    slab = min(remaining, 200)
    bill += slab * 4.5
    remaining -= slab
    if remaining <= 0:
        return bill

    # 401–500
    slab = min(remaining, 100)
    bill += slab * 6
    remaining -= slab
    if remaining <= 0:
        return bill

    # 501–600
    slab = min(remaining, 100)
    bill += slab * 8
    remaining -= slab
    if remaining <= 0:
        return bill

    # 601–800
    slab = min(remaining, 200)
    bill += slab * 9
    remaining -= slab
    if remaining <= 0:
        return bill

    # 801–1000
    slab = min(remaining, 200)
    bill += slab * 10
    remaining -= slab
    if remaining <= 0:
        return bill

    # Above 1000
    bill += remaining * 11

    return bill
# =====================================
# HEALTH CHECK
# =====================================
@app.get("/health")
def health():
    return {"status": "✅ Backend running"}

# =====================================
# PREDICT API (FIXED)
# =====================================
@app.post("/predict")
def predict(data: PredictionInput):

    if MODEL is None or SCALER is None:
        return {"error": "Model or scaler not loaded"}

    try:
        # ===============================
        # STEP 1 — Feature creation
        # ===============================
        X = create_features(data.appliances)
        kitchen, laundry, heavy, hours = X[0]

        manual_daily = kitchen + laundry + heavy

        # 🚨 no appliance safety
        if manual_daily <= 0:
            return {
                "daily_units": 0,
                "monthly_units": 0,
                "estimated_bill": 0
            }

        # ===============================
        # STEP 2 — Scale
        # ===============================
        X_scaled = SCALER.transform(X)

        # ===============================
        # STEP 3 — AI raw prediction
        # ===============================
        raw_ai = float(
            MODEL.predict(X_scaled, verbose=0)[0][0]
        )

        # ===============================
        # STEP 4 — Stable AI correction
        # ===============================
        ai_correction = np.tanh(raw_ai) * 0.15

        # ===============================
        # STEP 5 — Hybrid energy
        # ===============================
        daily_units = manual_daily * (1 + ai_correction)

        # ✅ PHYSICAL LIMITS
        daily_units = np.clip(
            daily_units,
            manual_daily * 0.7,
            manual_daily * 1.5
        )

        # ===============================
        # STEP 6 — Monthly
        # ===============================
        monthly_units = daily_units * 30

        # ✅ realistic household floor
        monthly_units = max(monthly_units, 120)

        # ===============================
        # STEP 7 — Bill
        # ===============================
        bill = tneb_bill(monthly_units)

        # ✅ UI stability (avoid ₹0)
        if bill <= 0:
            bill = monthly_units * 2.25

        return {
            "manual_daily_units": round(manual_daily, 2),
            "ai_correction_%": round(ai_correction * 100, 2),
            "daily_units": round(daily_units, 2),
            "monthly_units": round(monthly_units, 2),
            "estimated_bill": round(bill, 2)
        }

    except Exception as e:
        print("❌ Prediction error:", e)
        return {"error": str(e)}
# =====================================
# CHAT API
# =====================================
@app.post("/chat")
async def chat(data: ChatInput):
    if not data.message or not data.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = _normalize_azure_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT"))
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not api_key or not endpoint or not deployment:
        raise HTTPException(status_code=500, detail="Missing Azure OpenAI env config")

    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    headers = {"Content-Type": "application/json", "api-key": api_key}
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Hello! Ready to save energy? Let me know your needs—home, office, appliances, "
                    "or anything else—and I’ll share practical tips and efficient device suggestions.\n\n"
                    "Style rules:\n"
                    "- Be friendly and simple.\n"
                    "- Keep responses short (4-6 lines).\n"
                    "- Give actionable tips with estimated savings when possible.\n"
                    "- Suggest efficient alternatives (LED, BLDC fan, inverter AC, 5-star appliances).\n"
                    "- Use light emojis for readability."
                )
            },
            {"role": "user", "content": data.message}
        ],
        "temperature": 0.6,
        "max_tokens": 180
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, headers=headers, json=payload)

        if r.status_code != 200:
            # return Azure error to help debug
            raise HTTPException(status_code=502, detail=f"Azure error {r.status_code}: {r.text[:500]}")

        body = r.json()
        reply = body["choices"][0]["message"]["content"]
        return {"reply": reply}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failure: {str(e)}")

def _normalize_azure_endpoint(endpoint: str) -> str:
    ep = (endpoint or "").strip().rstrip("/")
    # If user put cognitiveservices endpoint, convert to OpenAI endpoint format
    if ".cognitiveservices.azure.com" in ep and ".openai.azure.com" not in ep:
        ep = ep.replace(".cognitiveservices.azure.com", ".openai.azure.com")
    return ep

# =====================================
# RUN SERVER
# =====================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
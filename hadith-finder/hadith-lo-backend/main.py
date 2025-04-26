from fastapi import FastAPI, HTTPException, Request, Depends, Header
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
from crew.tasks import run_crew
import traceback
from langdetect import detect, DetectorFactory
from mangum import Mangum

# Ensure consistent language detection
DetectorFactory.seed = 0

# Load .env file
load_dotenv()

# Get API keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in environment variables")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # For local testing
        "http://localhost:5173",  # For Vite local dev (if applicable)
        "https://your-frontend-domain.vercel.app"  # Replace with your frontend URL after deployment
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Pydantic model for request body
class QuestionRequest(BaseModel):
    question: str

# API key dependency
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

@app.post("/ask", dependencies=[Depends(verify_api_key)])
async def ask_question(req: QuestionRequest):
    try:
        user_input = req.question
        # Detect language
        try:
            lang = detect(user_input)
        except:
            lang = "en"
        result = run_crew(user_input)
        if not result:
            raise HTTPException(status_code=500, detail="No response from crew")
        return {"answer": result, "language": lang}
    except Exception as e:
        print(f"Error in /ask endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/debug")
async def debug():
    return {"message": "Backend is running"}

# Wrap FastAPI app with Mangum for Vercel serverless
handler = Mangum(app, lifespan="off")
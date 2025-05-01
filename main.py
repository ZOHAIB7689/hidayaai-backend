from fastapi import FastAPI, HTTPException, Request, Depends, Header
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
from crew.tasks import run_crew
import traceback
from langdetect import detect, DetectorFactory
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure consistent language detection
DetectorFactory.seed = 0

# Load .env file
load_dotenv()

# Get API keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables.")
    exit(1)

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logger.error("API_KEY not found in environment variables.")
    exit(1)

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing
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
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}")
            lang = "en"
        # Run crew with error handling
        try:
            result = run_crew(user_input)
            if not result:
                raise HTTPException(status_code=500, detail="No response from crew")
            return {"answer": result, "language": lang}
        except Exception as e:
            logger.error(f"Error in run_crew: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    except Exception as e:
        logger.error(f"Error in /ask endpoint: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/debug")
async def debug():
    return {"message": "Backend is running"}
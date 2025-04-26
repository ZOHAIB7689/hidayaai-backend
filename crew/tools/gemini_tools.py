import os
from dotenv import load_dotenv
import google.generativeai as genai
from crewai.tools import tool
from collections import OrderedDict
from collections import OrderedDict
from langdetect import detect , DetectorFactory

DetectorFactory.seed = 0
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# In-memory cache with size limit
cache = OrderedDict()
CACHE_SIZE = 1000

@tool("GeminiTool")
def gemini_tool(input: str) -> str:
    """Uses Gemini API to generate a response in the same language as the input."""
    try:
        # Check cache
        if input in cache:
            return cache[input]

        # Detect language
        try:
            lang = detect(input)
        except:
            lang = "en"  # Fallback to English if detection fails

        # Modify input to include language instruction
        if "categorize" in input.lower():
            input = f"{input}\nFor broad Islamic topics (e.g., 'fasting in Islam', 'prayer in Islam'), categorize as 'General Islamic' unless clearly specific to Hadith, Ayah, or stories. Respond in {lang}."
        else:
            input = f"{input}\nRespond in the same language as the input ({lang})."

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(input)
        result = response.text

        # Store in cache
        cache[input] = result
        if len(cache) > CACHE_SIZE:
            cache.popitem(last=False)
        return result
    except Exception as e:
        print(f"Error in Gemini API call: {e}")
        return f"Error: {str(e)}"
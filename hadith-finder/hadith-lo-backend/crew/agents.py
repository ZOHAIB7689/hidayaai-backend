import os
from dotenv import load_dotenv
from crewai import Agent
from crewai.llm import LLM
from crew.tools.gemini_tools import gemini_tool

# Load environment variables
load_dotenv()

# Configure Gemini LLM
gemini_llm = LLM(
    model="gemini/gemini-2.0-flash",  # Use Gemini model (litellm format)
    api_key=os.getenv("GEMINI_API_KEY"),  # Use Gemini API key
)

# Hadith Question Analyzer Agent
question_categorizer = Agent(
    role="Question Categorizer",
    goal="Analyze the user's question and categorize it as Hadith, Ayah, Islamic story, or non-Islamic",
    backstory="Expert in Islamic terminology and context, capable of identifying the nature of religious queries",    
    tools=[gemini_tool],
    llm=gemini_llm,  # Explicitly set the LLM to Gemini
    verbose=True
)

# Hadith Searcher Agent (optional, not used yet)
hadith_handler = Agent(
    role="Hadith  Expert",
    goal="Verify the authenticity of a Hadith, provide its full text, and explain its context and perspectives",
    backstory="Scholar with deep knowledge of Hadith sciences, incuding chains of narration and authenticity.",
    tools=[gemini_tool],
    llm=gemini_llm,  # Use the same LLM for consistency
    verbose=True
)

# Ayah Handler Agent
ayah_handler = Agent(
    role="Quranic Scholar",
    goal="Provide the translation, Shan-e-Nuzool, and context of a Quranic Ayah",
    backstory="Expert in Tafsir and Quranic studies, familiar with the context of revelation and interpretations",
    tools=[gemini_tool],
    llm=gemini_llm,
    verbose=True
)

# General Islamic Handler Agent
general_islamic_handler = Agent(
    role="Islamic Knowledge Expert",
    goal="Provide an overview of broad Islamic topics, referencing Hadiths, Ayahs, or stories as relevant, and guide users to clarify specific questions",
    backstory="Versatile scholar with comprehensive knowledge of Islamic teachings, capable of addressing general queries and providing context",
    tools=[gemini_tool],
    llm=gemini_llm,
    verbose=True
)
# Islamic Story Handler Agent
islamic_story_handler = Agent(
    role="Islamic Historian",
    goal="Share Islamic stories, their historical context, and significance",
    backstory="Knowledgeable in Islamic history and narratives, able to connect stories to their cultural and religious importance",
    tools=[gemini_tool],
    llm=gemini_llm,
    verbose=True
)

# Non-Islamic Response Agent
non_islamic_handler = Agent(
    role="General Responder",
    goal="Handle non-Islamic queries with appropriate responses or redirection",
    backstory="Versatile in handling diverse topics, providing polite and relevant responses",
    tools=[gemini_tool],
    llm=gemini_llm,
    verbose=True
)

# Debug print to confirm LLM configuration
# print(f"Question Categorizer LLM: {question_categorizer.llm}")
# print(f"Hadith Handler LLM: {hadith_handler.llm}")
# print(f"Ayah Handler LLM: {ayah_handler.llm}")
# print(f"Islamic Story Handler LLM: {islamic_story_handler.llm}")
# print(f"Non-Islamic Handler LLM: {non_islamic_handler.llm}")
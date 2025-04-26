from crewai import Task, Crew
from .agents import question_categorizer, hadith_handler, ayah_handler, islamic_story_handler, general_islamic_handler, non_islamic_handler
from langdetect import detect, DetectorFactory

# Ensure consistent language detection
DetectorFactory.seed = 0

# In-memory cache for task outputs
task_cache = {}

def run_crew(user_input: str):
    # Check cache
    if user_input in task_cache:
        return task_cache[user_input]

    # Detect language
    try:
        lang = detect(user_input)
    except:
        lang = "en"  # Fallback to English

    # Task 1: Categorize the question
    categorize_task = Task(
        description=f"Analyze the question '{user_input}' and categorize it as one of: Hadith, Ayah, Islamic story, General Islamic, or non-Islamic. If ambiguous, prioritize General Islamic for broad Islamic topics (e.g., 'fasting in Islam', 'prayer in Islam') or Islamic story for prophetic narratives (e.g., 'moses', 'yusuf'). For comparative religion questions (e.g., involving Christianity), prioritize non-Islamic. Return the category as a single word in English (e.g., 'Hadith', 'Ayah', 'Story', 'General', 'Non-Islamic'). Respond in {lang}.",
        expected_output="A single word indicating the category (Hadith, Ayah, Story, General, Non-Islamic)",
        agent=question_categorizer
    )

    # Define conditional tasks
    tasks_map = {
        "Hadith": Task(
            description=f"Verify the authenticity of the Hadith related to '{user_input}', provide its full text, and explain its context and perspectives. Respond in {lang}.",
            expected_output="Authenticity status, full text, context, and perspectives",
            agent=hadith_handler
        ),
        "Ayah": Task(
            description=f"Provide the translation, Shan-e-Nuzool, and context of the Quranic Ayah related to '{user_input}'. Respond in {lang}.",
            expected_output="Translation, Shan-e-Nuzool, and context",
            agent=ayah_handler
        ),
        "Story": Task(
            description=f"Share the story related to '{user_input}', its historical context, and significance. Respond in {lang}.",
            expected_output="Story details, historical context, and significance",
            agent=islamic_story_handler
        ),
        "General": Task(
            description=f"Provide a detailed overview of the topic '{user_input}', including its significance in Islam, relevant Quranic verses, Hadiths, and practical applications. Suggest clarifying questions like 'Would you like specific Hadiths or Quranic verses about {user_input}?' Respond in {lang}.",
            expected_output="Detailed overview with Quranic verses, Hadiths, and suggestions",
            agent=general_islamic_handler
        ),
        "Non-Islamic": Task(
            description=f"Provide an appropriate response or redirection for '{user_input}'. For comparative religion questions, compare the Islamic perspective with the other religion accurately. Respond in {lang}.",
            expected_output="Appropriate response or redirection",
            agent=non_islamic_handler
        )
    }

    # Run categorization first
    categorize_crew = Crew(
        agents=[question_categorizer],
        tasks=[categorize_task],
        verbose=False
    )

    try:
        categorize_result = categorize_crew.kickoff()
        category = categorize_result.raw if hasattr(categorize_result, 'raw') else str(categorize_result)
        category = category.strip()

        if category not in tasks_map:
            final_output = f"Sorry, I couldn't categorize your query. Please provide more details or ask about Hadith, Ayah, Islamic stories, or general Islamic topics. (Responded in {lang})"
        else:
            # Run only the relevant task
            relevant_task = tasks_map[category]
            crew = Crew(
                agents=[relevant_task.agent],
                tasks=[relevant_task],
                verbose=False
            )
            result = crew.kickoff()
            final_output = result.raw if hasattr(result, 'raw') else str(result)

            if not final_output or final_output == "Not applicable":
                final_output = f"Sorry, I couldn't find relevant information for your query. Please provide more details or ask about Hadith, Ayah, Islamic stories, or general Islamic topics. (Responded in {lang})"

        # Cache the result
        task_cache[user_input] = final_output
        return final_output.strip()
    except Exception as e:
        print(f"Error in run_crew: {str(e)}")
        raise
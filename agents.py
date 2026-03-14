import os
import time
import numpy as np
from dotenv import load_dotenv
from google import genai

# Load API key
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

MODEL = "gemini-2.5-flash"


# Retry wrapper for Gemini calls
def generate_with_retry(prompt):

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )
            return response.text

        except Exception as e:
            print(f"Model busy... retrying ({attempt+1}/3)")
            time.sleep(3)

    return "Model unavailable right now. Please try again later."


# Search agent
def search_agent(query):

    prompt = f"""
    Find 3 realistic Data Science / Machine Learning jobs in India for:

    {query}

    For each job provide:
    1. Job Title
    2. Company
    3. Required Skills
    4. Experience Level
    5. Short Job Description

    Keep it clean and readable.
    """

    return generate_with_retry(prompt)


# Resume analyzer agent
def resume_agent(resume_text):

    prompt = f"""
    Analyze this resume and extract:

    1. Key skills
    2. Experience level
    3. Strengths
    4. Missing skills
    5. Resume quality feedback

    Resume:
    {resume_text}

    Keep the response structured and recruiter-friendly.
    """

    return generate_with_retry(prompt)


# Skill Gap Agent
def skill_gap_agent(resume_text, jobs_text):

    prompt = f"""
    Compare the candidate resume with the target jobs.

    RESUME:
    {resume_text}

    TARGET JOBS:
    {jobs_text}

    Give a structured response with:
    1. Top matching skills
    2. Missing skills / skill gaps
    3. Important tools/technologies missing
    4. Resume improvement suggestions
    5. Priority order of what to learn first

    Make the output practical and specific for a fresher / entry-level candidate.
    """

    return generate_with_retry(prompt)


# Learning Roadmap Agent
def learning_roadmap_agent(resume_text, jobs_text):

    prompt = f"""
    Based on the candidate resume and the target jobs, create a practical 30-day learning roadmap.

    RESUME:
    {resume_text}

    TARGET JOBS:
    {jobs_text}

    Provide:
    1. Week 1 focus
    2. Week 2 focus
    3. Week 3 focus
    4. Week 4 focus
    5. Best projects to build
    6. Resume update suggestions after learning

    Keep it realistic for a student with limited time.
    """

    return generate_with_retry(prompt)


# Embedding Function
def get_embedding(text):

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return np.array(response.embeddings[0].values)


# Cosine Similarity
def cosine_similarity(vec1, vec2):

    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )
import os
import time
import numpy as np
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")


# Retry wrapper
def generate_with_retry(prompt):
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            return response.text

        except Exception as e:
            print(f"Gemini call failed (attempt {attempt+1}/3): {e}")
            time.sleep(2)

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


# Resume analysis
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


# Skill gap detector
def skill_gap_agent(resume_text, jobs_text):
    prompt = f"""
    Compare the candidate resume with the target jobs.

    RESUME:
    {resume_text}

    TARGET JOBS:
    {jobs_text}

    Give:
    1. Matching skills
    2. Missing skills
    3. Important tools missing
    4. Resume improvement suggestions
    5. Learning priorities

    Keep it practical for freshers.
    """

    return generate_with_retry(prompt)


# Learning roadmap
def learning_roadmap_agent(resume_text, jobs_text):
    prompt = f"""
    Based on the resume and target jobs, create a practical 30-day roadmap.

    RESUME:
    {resume_text}

    TARGET JOBS:
    {jobs_text}

    Include:
    1. Week 1
    2. Week 2
    3. Week 3
    4. Week 4
    5. Projects to build
    6. Resume improvements

    Keep it realistic for students.
    """

    return generate_with_retry(prompt)


# Embedding function
def get_embedding(text):
    try:
        result = genai.embed_content(
            model="models/gemini-embedding-001", # Updated to the current model
            content=text
        )
        
        return np.array(result["embedding"])
    except Exception as e:
        print(f"Embedding failed: {e}")
        return None


# Cosine similarity
def cosine_similarity(vec1, vec2):

    if vec1 is None or vec2 is None:
        return 0.0

    denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    if denominator == 0:
        return 0.0

    return np.dot(vec1, vec2) / denominator
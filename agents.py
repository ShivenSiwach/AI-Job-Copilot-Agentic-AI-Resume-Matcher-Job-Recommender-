import time
import numpy as np

MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"


# Retry wrapper for Gemini calls
def generate_with_retry(client, prompt, retries=3, delay=3):
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )
            return response.text

        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return "Model unavailable right now. Please try again later."


# Search agent
def search_agent(client, query):
    prompt = f"""
    Find 3 realistic entry-level Data Science / Machine Learning / Data Analyst jobs relevant to this query:

    {query}

    For each job provide:
    1. Job Title
    2. Company
    3. Required Skills
    4. Experience Level
    5. Short Job Description

    Keep it clean, realistic, and recruiter-friendly.
    """
    return generate_with_retry(client, prompt)


# Resume analyzer agent
def resume_agent(client, resume_text):
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
    return generate_with_retry(client, prompt)


# Skill Gap Agent
def skill_gap_agent(client, resume_text, jobs_text):
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
    return generate_with_retry(client, prompt)


# Learning Roadmap Agent
def learning_roadmap_agent(client, resume_text, jobs_text):
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
    return generate_with_retry(client, prompt)


# Embedding Function
def get_embedding(client, text):
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )

    return np.array(response.embeddings[0].values)


# Cosine Similarity
def cosine_similarity(vec1, vec2):
    denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    if denominator == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / denominator)
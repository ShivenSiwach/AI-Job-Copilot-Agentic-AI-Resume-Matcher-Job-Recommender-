import os
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")

# Search agent

def search_agent(query):
    prompt = f"""
    Find 3 realistic Data Science / ML jobs in India for:

    {query}

    For each job provide:
    - Company Name
    - Role
    - Required Skills
    - Experience Level
    """

    response = model.generate_content(prompt)
    return response.text



# Summarizer agent

def summarizer_agent(job_text):
    prompt = f"""
    Summarize the following job information.

    Extract clearly:
    - Key Technical Skills
    - Experience Required
    - Tools / Tech Stack

    Job Info:
    {job_text}
    """

    response = model.generate_content(prompt)
    return response.text



#  Critic agent

def critic_agent(job_summary):
    prompt = f"""
    Compare this job summary with a resume that has:

    Skills: Python, Machine Learning, SQL, Power BI

    Provide:
    - Match Score (0-100%)
    - Missing Skills
    - Suggestions to Improve Resume

    Job Summary:
    {job_summary}
    """

    response = model.generate_content(prompt)
    return response.text



# Embedding function

def get_embedding(text):
    result = genai.embed_content(
        model="models/embedding-001",
        content=text
    )
    return np.array(result["embedding"])


# COSINE SIMILARITY

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )
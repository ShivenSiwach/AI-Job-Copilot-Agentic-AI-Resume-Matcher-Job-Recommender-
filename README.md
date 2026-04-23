# AI Job Copilot: Agentic AI Resume Matcher and Job Recommender

AI Job Copilot is an AI-powered career intelligence application designed to help users analyze resumes, match them against job descriptions, identify skill gaps, and generate personalized learning roadmaps.

Built with Python, Streamlit, and Google Gemini API, the project combines resume parsing, semantic similarity, skill-gap detection, and LLM-driven career guidance into a practical end-to-end workflow. It is designed as a real-world portfolio project for students, freshers, and early-career professionals who want actionable career insights instead of generic resume feedback.

---

## Overview

AI Job Copilot helps users answer key career questions such as:

- Does my resume match this job?
- Which skills am I missing?
- What roles am I best suited for?
- What should I learn next to improve my chances?

The application provides a structured AI workflow that:

1. Accepts a resume upload (PDF)
2. Extracts and processes resume text
3. Compares resume content against job descriptions
4. Calculates semantic relevance / match score
5. Detects missing skills and improvement areas
6. Recommends suitable job roles
7. Generates a personalized learning roadmap

This project is well suited for portfolios targeting:

- AI Associate
- Data Analyst
- Machine Learning Intern
- NLP Intern
- Python Developer
- GenAI / LLM Intern
- Career-tech / HR-tech roles

---

## Key Features

### Resume Upload and Parsing
- Upload resumes in PDF format
- Extract text from resumes for downstream analysis
- Prepare structured content for semantic comparison

### Job Description Analysis
- Accept job descriptions as user input
- Normalize and process role requirements
- Extract relevant keywords and skill signals

### Semantic Resume-to-Job Matching
- Compare resume content with job descriptions using embeddings
- Compute a semantic relevance / similarity score
- Move beyond simple keyword matching

### Skill Gap Detection
- Identify missing or underrepresented skills
- Highlight important capabilities required by the target role
- Provide actionable improvement areas

### Role Recommendation
- Suggest suitable roles based on resume content
- Help users who are unsure which roles best match their current profile

### Personalized Learning Roadmap
- Generate a structured learning plan (for example, a 30-day roadmap)
- Recommend what to learn next based on skill gaps
- Convert career advice into practical next steps

### Agentic / Modular AI Workflow
- Organize the solution into logical modules such as:
  - Resume parsing
  - Resume analysis
  - Job matching
  - Skill-gap detection
  - Roadmap generation

### Interactive Web Interface
- Built with Streamlit for quick and user-friendly interaction
- Suitable for demos, portfolio showcases, and future deployment

---

## Problem Statement

Many students and job seekers face the same challenges:

- They do not know whether their resume actually fits a role
- They are unsure which roles align with their skills
- They do not know what skills are missing
- They receive generic career advice instead of targeted feedback

AI Job Copilot addresses these problems by combining:

- Resume analysis
- Job-role alignment
- Skill-gap intelligence
- Personalized learning guidance

---

## Tech Stack

### Core Language
- Python

### Frontend / Application Framework
- Streamlit

### AI / NLP / LLM
- Google Gemini API
- Embeddings
- Cosine Similarity
- Prompt-based analysis
- NLP preprocessing

### Data Processing
- Pandas
- NumPy

### Resume Parsing / File Handling
- PyPDF / PDF parsing utilities

### Optional Supporting Libraries
- Scikit-learn (if used for similarity support or vectorization)
- Regex
- JSON
- Python-dotenv

---

## Recommended Project Structure

> If your current repository uses a simpler structure (for example, a single `app.py`), that is fine. This is the recommended professional structure for future improvement.

```bash
AI-Job-Copilot-Agentic-AI-Resume-Matcher-Job-Recommender-/
│── app.py
│── requirements.txt
│── README.md
│── .gitignore
│── .env.example
│
├── modules/
│   ├── resume_parser.py
│   ├── resume_analyzer.py
│   ├── job_matcher.py
│   ├── skill_gap_analyzer.py
│   ├── role_recommender.py
│   ├── roadmap_generator.py
│   └── utils.py
│
├── assets/
│   └── screenshots/
│
├── data/
│   ├── sample_jobs.csv
│   └── sample_resumes/
│
└── temp_uploads/

from agents import (search_agent,summarizer_agent, get_embedding, cosine_similarity)

def main():

    query = "Data Scientist with 0-2 years experience in India"

    print("🔎 Searching Jobs...\n")
    jobs = search_agent(query)
    print(jobs)

    print("\n📄 Summarizing Jobs...\n")
    summary = summarizer_agent(jobs)
    print(summary)
    
    # Embedding-Based Similarity
    resume_text = "Python, Machine Learning, SQL, Power BI"

    resume_embedding = get_embedding(resume_text)
    job_embedding = get_embedding(summary)

    similarity = cosine_similarity(resume_embedding, job_embedding)

    print("\n Embedding Match Score:")
    print(f"{round(similarity * 100, 2)}%")


if __name__ == "__main__":
    main()
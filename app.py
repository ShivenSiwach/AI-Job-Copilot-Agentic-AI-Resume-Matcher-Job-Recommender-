import streamlit as st
from pypdf import PdfReader
from agents import (search_agent,resume_agent,skill_gap_agent,learning_roadmap_agent,get_embedding,cosine_similarity)


# Page config

st.set_page_config(
    page_title="AI Job Copilot",
    page_icon="",
    layout="wide"
)


# Helper: Extract text from PDF

def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_reader = PdfReader(uploaded_file)

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()


# Header

st.title(" AI Job Copilot")
st.subheader("AI Resume Matcher + Skill Gap Detector + Career Coach")

st.markdown("---")


# Sidebar

st.sidebar.header(" Job Search Settings")

job_query = st.sidebar.text_input(
    "Enter Job Query",
    value="Entry level Data Scientist with Python and Machine Learning"
)

st.sidebar.markdown("### Example Queries")
st.sidebar.write("- Data Analyst with SQL and Power BI")
st.sidebar.write("- Machine Learning Intern")
st.sidebar.write("- Junior Data Scientist")
st.sidebar.write("- Data Scientist Fresher")

st.sidebar.markdown("---")
st.sidebar.info("Upload your resume PDF and get AI-powered job matching + career guidance.")


# Upload Section

st.header(" Upload Resume PDF")

uploaded_file = st.file_uploader(
    "Upload your resume in PDF format",
    type=["pdf"]
)

resume_text = ""

if uploaded_file is not None:
    try:
        resume_text = extract_text_from_pdf(uploaded_file)
        st.success(" Resume PDF uploaded and text extracted successfully!")

        with st.expander(" Preview Extracted Resume Text"):
            st.text_area("Extracted Text", resume_text, height=250)

    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")


# Analyze Button

analyze_button = st.button(" Analyze Resume & Build Career Plan")

if analyze_button:
    if not uploaded_file:
        st.warning(" Please upload a resume PDF first.")
    elif not resume_text.strip():
        st.warning(" Resume text could not be extracted from the PDF.")
    else:
        try:
            # Step 1: Jobs
            with st.spinner(" Searching for matching jobs..."):
                jobs = search_agent(job_query)

            # Step 2: Resume Analysis
            with st.spinner(" Analyzing resume..."):
                analysis = resume_agent(resume_text)

            # Step 3: Skill Gap
            with st.spinner(" Detecting skill gaps..."):
                skill_gap = skill_gap_agent(resume_text, jobs)

            # Step 4: Roadmap
            with st.spinner(" Building 30-day learning roadmap..."):
                roadmap = learning_roadmap_agent(resume_text, jobs)

            # Step 5: Embedding Score
            with st.spinner(" Calculating semantic match score..."):
                job_embedding = get_embedding(jobs)
                resume_embedding = get_embedding(resume_text)
                score = cosine_similarity(job_embedding, resume_embedding)

            match_percent = round(score * 100, 2)

            st.success(" Analysis completed successfully!")

            st.markdown("---")

            # Tabs for better UX
            tab1, tab2, tab3, tab4 = st.tabs([
                " Jobs Found",
                " Resume Analysis",
                " Skill Gap Report",
                "30-Day Roadmap"
            ])

            with tab1:
                st.subheader(" Recommended Jobs")
                st.write(jobs)

            with tab2:
                st.subheader(" Resume Analysis")
                st.write(analysis)

            with tab3:
                st.subheader(" Skill Gap Detector")
                st.write(skill_gap)

            with tab4:
                st.subheader(" 30-Day Learning Roadmap")
                st.write(roadmap)

            st.markdown("---")

            # Match score section
            st.subheader("Resume-Job Match Score")
            st.metric(label="Match Score", value=f"{match_percent}%")
            st.progress(min(max(match_percent / 100, 0.0), 1.0))

            # Score interpretation
            if match_percent >= 80:
                st.success(" Excellent match! Your resume is strongly aligned.")
            elif match_percent >= 60:
                st.info(" Good match! Some improvements can increase alignment.")
            else:
                st.warning(" Low match. Improve your resume and target skills.")

        except Exception as e:
            st.error(f" Something went wrong: {str(e)}")

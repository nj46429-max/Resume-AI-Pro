import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

def preprocess(text):
    text = text.lower()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words and w not in string.punctuation]
    return " ".join(words)

def extract_keywords(text):
    return set(text.lower().split())

st.title("ResumeAI Pro - AI Resume Planner")

resume = st.text_area("Paste Resume Here")
job_desc = st.text_area("Paste Job Description Here")

if st.button("Analyze Resume"):
    if resume == "" or job_desc == "":
        st.warning("Enter both fields")
    else:
        resume_clean = preprocess(resume)
        jd_clean = preprocess(job_desc)

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_clean, jd_clean])

        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        score = round(similarity * 100, 2)

        st.subheader(f"Match Score: {score}%")

        jd_keywords = extract_keywords(job_desc)
        resume_keywords = extract_keywords(resume)

        missing = jd_keywords - resume_keywords

        st.subheader("Missing Skills:")
        st.write(list(missing)[:10])

        if score < 50:
            st.error("Low match. Improve skills.")
        elif score < 75:
            st.warning("Moderate match.")
        else:
            st.success("Good match!")
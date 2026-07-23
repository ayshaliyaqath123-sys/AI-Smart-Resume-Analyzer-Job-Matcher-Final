"""
job_matcher.py
Compares a resume against a set of job postings (or a single pasted job
description) using TF-IDF vectors + cosine similarity.

TF-IDF is used instead of Sentence-Transformers so the project runs with
no large model downloads. Swapping in `sentence-transformers` later is a
drop-in change - see README "Next Steps".
"""
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "jobs_db.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    JOBS = json.load(f)


def recommend_jobs(resume_text: str, top_n: int = 4) -> list:
    """Rank the jobs_db.json postings by similarity to the resume text."""
    job_texts = [job["description"] for job in JOBS]
    corpus = [resume_text] + job_texts

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    resume_vec = tfidf_matrix[0:1]
    job_vecs = tfidf_matrix[1:]
    similarities = cosine_similarity(resume_vec, job_vecs).flatten()

    ranked = sorted(
        zip(JOBS, similarities), key=lambda pair: pair[1], reverse=True
    )[:top_n]

    return [
        {"title": job["title"], "match_score": round(float(sim) * 100, 1)}
        for job, sim in ranked
    ]


def match_against_description(resume_text: str, job_description: str, master_skills: list) -> dict:
    """Compare a resume directly against a pasted job description:
    returns an overall similarity score plus skill keywords present in the
    JD but missing from the resume."""
    corpus = [resume_text, job_description]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    jd_lower = job_description.lower()
    resume_lower = resume_text.lower()

    missing_keywords = []
    for skill in master_skills:
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9])"
        in_jd = bool(re.search(pattern, jd_lower))
        in_resume = bool(re.search(pattern, resume_lower))
        if in_jd and not in_resume:
            missing_keywords.append(skill)

    return {
        "match_score": round(float(similarity) * 100, 1),
        "missing_keywords": missing_keywords,
    }

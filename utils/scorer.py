"""
scorer.py
Rule-based scoring of a resume across several dimensions:
Skills, Education, Projects, Experience, Formatting, and an overall ATS score.

This is heuristic (not a black box ML model) on purpose - it's transparent,
explainable, and doesn't need training data to get started. You can later
replace individual checks with smarter NLP without changing the interface.
"""
import re

EDUCATION_KEYWORDS = [
    "bachelor", "b.tech", "btech", "b.e.", "bsc", "b.sc", "master", "m.tech",
    "mtech", "msc", "m.sc", "mba", "phd", "degree", "university", "college",
    "institute of technology"
]

EXPERIENCE_KEYWORDS = ["experience", "internship", "intern", "worked at", "employed"]
PROJECT_KEYWORDS = ["project", "projects", "built", "developed", "implemented"]

SECTION_HEADERS = [
    "education", "experience", "skills", "projects", "certification",
    "summary", "objective", "achievements"
]


def _clamp(value, low=0, high=100):
    return max(low, min(high, value))


def score_skills(found_skills: list, expected_total: int = 12) -> int:
    """Score scales up to 100 as the resume approaches `expected_total` distinct skills."""
    if expected_total <= 0:
        return 0
    return int(_clamp((len(found_skills) / expected_total) * 100))


def score_education(text: str) -> int:
    text_lower = text.lower()
    hits = sum(1 for kw in EDUCATION_KEYWORDS if kw in text_lower)
    if hits == 0:
        return 30  # some education info might exist but wasn't detected
    return int(_clamp(50 + hits * 15))


def score_projects(text: str) -> int:
    text_lower = text.lower()
    hits = sum(text_lower.count(kw) for kw in PROJECT_KEYWORDS)
    return int(_clamp(hits * 12))


def score_experience(text: str) -> int:
    text_lower = text.lower()
    hits = sum(text_lower.count(kw) for kw in EXPERIENCE_KEYWORDS)

    years_match = re.findall(r"(\d+)\+?\s*(?:years|yrs)", text_lower)
    years_bonus = min(sum(int(y) for y in years_match), 5) * 10 if years_match else 0

    return int(_clamp(hits * 15 + years_bonus))


def score_formatting(text: str) -> int:
    score = 40  # baseline
    has_email = bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text))
    has_phone = bool(re.search(r"(\+?\d[\d\-\s]{8,}\d)", text))
    has_bullets = bool(re.search(r"[\u2022\-\*]\s", text))
    reasonable_length = 400 <= len(text) <= 8000

    if has_email:
        score += 15
    if has_phone:
        score += 15
    if has_bullets:
        score += 15
    if reasonable_length:
        score += 15

    return int(_clamp(score))


def score_ats(text: str) -> int:
    """Approximates ATS-friendliness: are standard section headers present,
    and is the text cleanly parseable (no huge blocks of garbled characters)."""
    text_lower = text.lower()
    headers_found = sum(1 for h in SECTION_HEADERS if h in text_lower)
    header_score = min(headers_found / len(SECTION_HEADERS), 1) * 60

    # Penalize very short extracted text -> often means the PDF was image-based
    # (scanned) or used complex multi-column layouts that confuse ATS parsers.
    length_score = 40 if len(text.strip()) > 300 else 10

    return int(_clamp(header_score + length_score))


def compute_resume_score(text: str, found_skills: list, category: str = None,
                          category_requirements: dict = None) -> dict:
    expected_total = 12
    if category and category_requirements:
        expected_total = max(len(category_requirements.get(category, [])), 6)

    skills = score_skills(found_skills, expected_total)
    education = score_education(text)
    projects = score_projects(text)
    experience = score_experience(text)
    formatting = score_formatting(text)
    ats = score_ats(text)

    overall = int(_clamp(
        skills * 0.30 + education * 0.15 + projects * 0.20 +
        experience * 0.15 + formatting * 0.10 + ats * 0.10
    ))

    return {
        "overall": overall,
        "skills": skills,
        "education": education,
        "projects": projects,
        "experience": experience,
        "formatting": formatting,
        "ats": ats,
    }

"""
skill_extractor.py
Simple, dependency-light skill detection using regex keyword matching
against a master skills list (data/skills_db.json).

NOTE: This is intentionally rule-based so the project works out of the box
with no model downloads. Once this is working, you can swap this module
for a spaCy NER pipeline or a Sentence-Transformers embedding match -
see the "Next Steps" section in README.md.
"""
import re
import json
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "skills_db.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    SKILLS_DATA = json.load(f)

MASTER_SKILLS = SKILLS_DATA["master_skills"]
CATEGORY_REQUIREMENTS = SKILLS_DATA["category_requirements"]


def extract_skills(resume_text: str) -> list:
    """Return the list of master skills found in the resume text."""
    found = []
    text_lower = resume_text.lower()

    for skill in MASTER_SKILLS:
        # Build a word-boundary pattern so "R" doesn't match inside "Robert"
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.append(skill)

    return found


def missing_skills_for_category(found_skills: list, category: str) -> list:
    """Compare found skills against what a target job category requires."""
    required = CATEGORY_REQUIREMENTS.get(category, [])
    found_set = set(found_skills)
    return [skill for skill in required if skill not in found_set]


def available_categories() -> list:
    return list(CATEGORY_REQUIREMENTS.keys())

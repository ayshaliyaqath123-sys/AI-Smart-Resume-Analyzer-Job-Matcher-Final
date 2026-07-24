"""
app.py
Flask entry point for the AI Smart Resume Analyzer & Job Matcher.

Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""
import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename

from utils.extractor import extract_text
from utils.skill_extractor import extract_skills, missing_skills_for_category, available_categories, MASTER_SKILLS
from utils.scorer import compute_resume_score
from utils.job_matcher import recommend_jobs, match_against_description
from utils.visualizer import generate_skill_gap_chart
from utils.chatbot import answer_question
from utils import classifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"pdf", "docx"}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


PORTFOLIO_SUGGESTIONS = {
    "Data Scientist": ["End-to-end ML pipeline", "NLP sentiment classifier", "Interactive Power BI dashboard"],
    "Machine Learning Intern": ["Image classifier (CNN)", "Kaggle competition submission", "ML model deployed as an API"],
    "Python Developer": ["REST API with Flask/Django", "CLI automation tool", "Dockerized microservice"],
    "Data Analyst": ["Sales dashboard in Power BI", "SQL data cleaning project", "A/B testing analysis"],
    "Business Analyst": ["Process automation case study", "Requirements documentation sample", "KPI dashboard"],
    "Software Engineer": ["Full-stack web app", "System design write-up", "Open-source contribution"],
    "Cloud Engineer": ["CI/CD pipeline demo", "Kubernetes deployment project", "Infrastructure-as-code with Terraform"],
    "UI Designer": ["Mobile app UI case study", "Design system in Figma", "Usability testing report"],
}


@app.route("/")
def index():
    return render_template("index.html", categories=available_categories())


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("resume")
    category = request.form.get("category", "")
    job_description = request.form.get("job_description", "").strip()

    if not file or file.filename == "":
        return render_template("index.html", categories=available_categories(),
                                error="Please choose a PDF or DOCX file to upload.")

    if not allowed_file(file.filename):
        return render_template("index.html", categories=available_categories(),
                                error="Only PDF and DOCX files are supported.")

    unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    try:
        resume_text = extract_text(filepath)
    except Exception as exc:
        return render_template("index.html", categories=available_categories(),
                                error=f"Could not read that file: {exc}")
    finally:
        # Don't keep uploaded resumes lying around on disk longer than needed
        if os.path.exists(filepath):
            os.remove(filepath)

    if not resume_text or len(resume_text.strip()) < 30:
        return render_template("index.html", categories=available_categories(),
                                error="Couldn't extract readable text from that file. "
                                      "If it's a scanned/image PDF, try a text-based export.")

    found_skills = extract_skills(resume_text)
    missing = missing_skills_for_category(found_skills, category) if category else []

    from utils.skill_extractor import CATEGORY_REQUIREMENTS
    scores = compute_resume_score(resume_text, found_skills, category, CATEGORY_REQUIREMENTS)

    jobs = recommend_jobs(resume_text)

    chart_data_uri = None
    if category:
        required = CATEGORY_REQUIREMENTS.get(category, [])
        chart_data_uri = generate_skill_gap_chart(required, found_skills)

    jd_match = None
    if job_description:
        jd_match = match_against_description(resume_text, job_description, MASTER_SKILLS)

    predicted_category = None
    if classifier.model_available():
        try:
            predicted_category = classifier.predict_category(resume_text)
        except Exception:
            predicted_category = None

    suggestions = build_suggestions(scores, missing)
    portfolio_ideas = PORTFOLIO_SUGGESTIONS.get(category, PORTFOLIO_SUGGESTIONS["Software Engineer"])

    # Store just enough context for the chatbot to reference later
    session["chat_context"] = {
        "scores": scores,
        "missing_skills": missing,
        "found_skills": found_skills,
        "category": category or "your target role",
    }

    return render_template(
        "results.html",
        scores=scores,
        found_skills=found_skills,
        missing_skills=missing,
        category=category,
        jobs=jobs,
        chart_data_uri=chart_data_uri,
        jd_match=jd_match,
        predicted_category=predicted_category,
        suggestions=suggestions,
        portfolio_ideas=portfolio_ideas,
    )


def build_suggestions(scores: dict, missing_skills: list) -> list:
    tips = []
    if scores["experience"] < 60:
        tips.append("Add measurable achievements to your experience section (numbers, %, outcomes).")
    if scores["projects"] < 60:
        tips.append("Include more detailed project descriptions - problem, approach, result.")
    if scores["formatting"] < 70:
        tips.append("Make sure your contact info, bullet points, and section headers are clear and consistent.")
    if scores["ats"] < 70:
        tips.append("Simplify formatting (avoid tables/columns/images) so ATS systems can parse your resume.")
    if missing_skills:
        tips.append(f"Consider learning/adding: {', '.join(missing_skills[:5])}.")
    tips.append("Include a link to your GitHub or portfolio.")
    tips.append("Mention any relevant certifications.")
    return tips


@app.route("/chat")
def chat_page():
    if "chat_context" not in session:
        return redirect(url_for("index"))
    return render_template("chat.html")


@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    context = session.get("chat_context", {})

    if not question:
        return jsonify({"answer": "Ask me something about your resume!"})

    answer = answer_question(question, context)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

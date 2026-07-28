"""
app.py
Flask entry point for the AI Smart Resume Analyzer & Job Matcher.

Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""
import os
import sqlite3
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.security import check_password_hash, generate_password_hash
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
DB_PATH = os.path.join(BASE_DIR, "users.db")
ALLOWED_EXTENSIONS = {"pdf", "docx"}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            user_name TEXT,
            category TEXT,
            overall_score REAL,
            skills_score REAL,
            education_score REAL,
            projects_score REAL,
            experience_score REAL,
            formatting_score REAL,
            ats_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_user_by_email(email: str):
    normalized_email = (email or "").strip().lower()
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE lower(email) = ?", (normalized_email,)).fetchone()
    if user is None:
        user = conn.execute("SELECT * FROM users WHERE email = ? COLLATE NOCASE", (normalized_email,)).fetchone()
    conn.close()
    return user


def create_user(name: str, email: str, password: str):
    conn = get_db()
    cursor = conn.execute(
        "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
        (name, email.lower(), generate_password_hash(password)),
    )
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return user_id


def verify_password(stored_password: str, provided_password: str) -> bool:
    if not stored_password:
        return False

    if stored_password == provided_password:
        return True

    try:
        return check_password_hash(stored_password, provided_password)
    except Exception:
        return False


def authenticate_user(email: str, password: str):
    user = get_user_by_email(email)
    if not user:
        return None

    stored_password = user["password"] or ""
    if not verify_password(stored_password, password):
        return None

    if stored_password == password:
        conn = get_db()
        conn.execute(
            "UPDATE users SET password = ? WHERE id = ?",
            (generate_password_hash(password), user["id"]),
        )
        conn.commit()
        conn.close()

    return user


def save_analysis_record(scores: dict, category: str) -> None:
    user = session.get("user") or {}
    user_email = user.get("email")
    user_name = user.get("name")

    conn = get_db()
    conn.execute(
        """
        INSERT INTO analysis_history (
            user_email, user_name, category, overall_score, skills_score,
            education_score, projects_score, experience_score,
            formatting_score, ats_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_email,
            user_name,
            category,
            scores.get("overall"),
            scores.get("skills"),
            scores.get("education"),
            scores.get("projects"),
            scores.get("experience"),
            scores.get("formatting"),
            scores.get("ats"),
        ),
    )
    conn.commit()
    conn.close()


init_db()


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


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        if not email or not password:
            return render_template("auth.html", mode="signin", error="Please enter your email and password.")

        user = authenticate_user(email, password)
        if not user:
            return render_template("auth.html", mode="signin", error="Invalid email or password.")

        session["user"] = {"email": user["email"], "name": user["name"]}
        return redirect(url_for("index"))

    return render_template("auth.html", mode="signin", current_user=session.get("user"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not name or not email or not password or not confirm_password:
            return render_template("auth.html", mode="signup", error="Please fill in all fields.")

        if password != confirm_password:
            return render_template("auth.html", mode="signup", error="Passwords do not match.")

        if get_user_by_email(email):
            return render_template("auth.html", mode="signup", error="An account with this email already exists.")

        create_user(name, email, password)
        session["user"] = {"email": email.lower(), "name": name}
        return redirect(url_for("index"))

    return render_template("auth.html", mode="signup", current_user=session.get("user"))


@app.route("/signout")
def signout():
    session.pop("user", None)
    return redirect(url_for("signin"))


@app.route("/profile")
def profile():
    if "user" not in session:
        return redirect(url_for("signin"))
    return render_template("profile.html", current_user=session.get("user"))


@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("signup"))
    return render_template("index.html", categories=available_categories(), current_user=session.get("user"))


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

    chart_data = None
    if category:
        required = CATEGORY_REQUIREMENTS.get(category, [])
        chart_data = generate_skill_gap_chart(required, found_skills)

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
    save_analysis_record(scores, category)

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
        chart_data=chart_data,
        jd_match=jd_match,
        predicted_category=predicted_category,
        suggestions=suggestions,
        portfolio_ideas=portfolio_ideas,
        current_user=session.get("user"),
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
    return render_template("chat.html", current_user=session.get("user"))


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

# AI Smart Resume Analyzer & Job Matcher

A Flask web app that scores a resume, detects skills, finds missing skills for
a target job, recommends matching jobs, compares against a pasted job
description, visualizes the skill gap, and answers questions through a chatbot.

This has been built and tested end-to-end (including a real trained ML model
and live HTTP requests) - it works out of the box.

## 1. What's inside

```
resume_analyzer/
├── app.py                  # Flask app - all routes
├── train_classifier.py     # Trains the resume-category ML model
├── requirements.txt
├── utils/
│   ├── extractor.py        # PDF/DOCX -> plain text
│   ├── skill_extractor.py  # Keyword-based skill detection
│   ├── scorer.py           # Resume scoring (skills/education/projects/...)
│   ├── job_matcher.py      # TF-IDF + cosine similarity job matching
│   ├── classifier.py       # Loads the trained ML model, predicts category
│   ├── chatbot.py          # Rule-based chatbot (OpenAI/Gemini hook included)
│   └── visualizer.py       # Matplotlib skill-gap chart -> base64 PNG
├── data/
│   ├── skills_db.json      # Master skill list + per-category requirements
│   ├── jobs_db.json        # Sample job postings used for recommendations
│   └── sample_resumes.csv  # Tiny synthetic dataset to train the classifier
├── models/                 # Trained model gets saved here (.joblib)
├── templates/               # HTML pages (Bootstrap 5)
└── static/                  # CSS + temporary upload folder
```

## 2. Setup (step by step)

**Step 1 - Install Python.** You need Python 3.10+ installed. Check with:
```bash
python3 --version
```

**Step 2 - Create a virtual environment** (keeps this project's packages separate
from everything else on your machine):
```bash
cd resume_analyzer
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
```

**Step 3 - Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 4 - Train the ML classifier** (only needs to be run once, or whenever
you update the training data):
```bash
python train_classifier.py
```
You'll see a small accuracy report and a `models/resume_classifier.joblib`
file get created. The included dataset is tiny (32 rows) and synthetic, so
accuracy will look mediocre - that's expected and explained in "Next Steps"
below.

**Step 5 - Run the app:**
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

**Step 6 - Try it.** Upload any PDF or DOCX resume, pick a target category
like "Data Scientist", optionally paste a job description, and click
"Analyze Resume". Then click "Ask the AI Chatbot" to ask questions about your
results.

## 3. How each feature maps to the code

| Feature (from your spec)       | Where it lives                          | How it works |
|---------------------------------|------------------------------------------|--------------|
| Resume upload + text extraction | `utils/extractor.py`                     | `pdfplumber` for PDF, `python-docx` for DOCX |
| Resume score                    | `utils/scorer.py`                        | Rule-based checks per dimension (regex + keyword counts), weighted average |
| Skill extraction                | `utils/skill_extractor.py`               | Regex keyword match against `data/skills_db.json` |
| Missing skills detection        | `utils/skill_extractor.py`               | Set difference: category requirements minus found skills |
| Job recommendation              | `utils/job_matcher.py` → `recommend_jobs`| TF-IDF vectors + cosine similarity vs `data/jobs_db.json` |
| Resume improvement suggestions  | `app.py` → `build_suggestions()`         | Rule-based tips triggered by low sub-scores |
| AI chatbot                      | `utils/chatbot.py`                       | Rule-based keyword matching using your actual score/skills as context |
| JD matching                     | `utils/job_matcher.py` → `match_against_description` | TF-IDF similarity + keyword diff between resume and pasted JD |
| Skill gap visualization         | `utils/visualizer.py`                    | Matplotlib horizontal bar chart, returned as an inline base64 PNG (no file saved) |
| Portfolio suggestions           | `app.py` → `PORTFOLIO_SUGGESTIONS`       | Static suggestions keyed by category |
| Resume category (ML)            | `train_classifier.py`, `utils/classifier.py` | TF-IDF + RandomForest trained on `data/sample_resumes.csv` |

## 4. Why some things are simplified (and how to level them up)

This is built to actually run on day one with no API keys, no GPU, and no
multi-gigabyte model downloads. A few intentional simplifications, and how to
upgrade each one once the basic app is working:

- **Skill extraction is regex/keyword-based, not spaCy NER.**
  It's reliable and needs zero downloads. To upgrade: `pip install spacy`,
  download a model (`python -m spacy download en_core_web_sm`), and replace
  the loop in `extract_skills()` with an NER + phrase-matcher pipeline
  (`spacy.matcher.PhraseMatcher` is a natural fit here).

- **Job/resume similarity uses TF-IDF, not Sentence-Transformers.**
  TF-IDF needs no model download and is fast. To upgrade:
  `pip install sentence-transformers`, encode resume/job text with
  `SentenceTransformer('all-MiniLM-L6-v2')`, and swap `cosine_similarity`
  to compare embeddings instead of TF-IDF vectors - the surrounding function
  signatures don't need to change.

- **The classifier's training data is a 32-row synthetic CSV.**
  It's enough to prove the pipeline works, not enough for real accuracy.
  Download a proper labeled dataset from Kaggle (search "resume dataset" or
  "resume classification"), reformat it to match the `resume_text,category`
  columns in `data/sample_resumes.csv`, and re-run `train_classifier.py`.

- **The chatbot is rule-based, not a real LLM.**
  It already uses your actual resume score and missing skills as context, so
  answers are personalized even without an API. To connect a real model: get
  an API key from OpenAI or Google AI Studio (Gemini), `pip install openai`
  (or `google-generativeai`), set `OPENAI_API_KEY` as an environment
  variable, and flip `USE_EXTERNAL_API = True` in `utils/chatbot.py` - the
  `call_external_api()` function already has the exact code commented in.

## 5. Common issues

- **"Trained model not found"** → run `python train_classifier.py` first.
- **"Couldn't extract readable text"** → the PDF is probably a scanned image
  rather than real text. Export it as text-based PDF, or add OCR
  (`pytesseract`) as a fallback in `extractor.py`.
- **Port already in use** → another process is using port 5000; run
  `app.run(debug=True, port=5001)` instead.
- **Uploaded resumes aren't saved anywhere** → that's intentional (privacy);
  the file is deleted right after its text is extracted.

## 6. Suggested order to learn/extend this project

1. Get it running locally exactly as-is (Steps 1-6 above).
2. Read `app.py` top to bottom - it's the map of how every feature connects.
3. Try changing `data/skills_db.json` - add your own skills/categories and
   see the app update immediately.
4. Swap in a real Kaggle dataset for `train_classifier.py`.
5. Upgrade skill extraction to spaCy, then similarity to
   Sentence-Transformers (see section 4).
6. Add a real LLM to the chatbot.
7. Deploy it (Render, Railway, or PythonAnywhere are the easiest free options
   for a Flask app like this).

"""
chatbot.py
A small rule-based chatbot that answers common resume questions using the
analysis already computed for the user's resume (score, missing skills,
category). This works with zero API keys.

If you want real generative answers, set OPENAI_API_KEY (or GEMINI_API_KEY)
as an environment variable and flip USE_EXTERNAL_API to True - the
call_external_api() function below shows exactly where to plug it in.
"""
import os

USE_EXTERNAL_API = False  # flip to True once you've added an API key


def answer_question(question: str, context: dict) -> str:
    """context is expected to contain: scores (dict), missing_skills (list),
    found_skills (list), category (str)."""
    q = question.lower()

    scores = context.get("scores", {})
    missing_skills = context.get("missing_skills", [])
    found_skills = context.get("found_skills", [])
    category = context.get("category", "your target role")

    if USE_EXTERNAL_API:
        return call_external_api(question, context)

    if "ats" in q or "why is my ats" in q or "ats score low" in q:
        ats = scores.get("ats", "N/A")
        return (
            f"Your ATS score is {ats}%. ATS parsers struggle with scanned/image PDFs, "
            "complex multi-column layouts, tables, and missing standard section headers "
            "(like 'Experience', 'Education', 'Skills'). Use a single-column layout with "
            "clear text-based headers and avoid embedding text inside images."
        )

    if "improve" in q or "better" in q:
        return (
            "A few concrete improvements: add measurable achievements (e.g. 'reduced "
            "processing time by 30%'), link your GitHub/portfolio, list relevant "
            "certifications, and make sure each project description explains the "
            f"problem, your approach, and the result. For {category}, also consider "
            f"adding: {', '.join(missing_skills[:5]) if missing_skills else 'you already cover the core skills - nice work!'}."
        )

    if "which project" in q or "what project" in q:
        return (
            f"Based on the skills you're missing for {category} "
            f"({', '.join(missing_skills[:5]) if missing_skills else 'none major'}), "
            "build a small end-to-end project that uses those tools and put it on GitHub "
            "with a clear README, screenshots, and a short write-up of your approach."
        )

    if "skill" in q and ("learn" in q or "missing" in q or "need" in q):
        if missing_skills:
            return f"For {category}, focus on learning: {', '.join(missing_skills)}."
        return f"You already cover the core skills typically required for {category}!"

    if "score" in q:
        overall = scores.get("overall", "N/A")
        return (
            f"Your overall resume score is {overall}%. Breakdown - "
            f"Skills: {scores.get('skills','N/A')}%, Education: {scores.get('education','N/A')}%, "
            f"Projects: {scores.get('projects','N/A')}%, Experience: {scores.get('experience','N/A')}%, "
            f"Formatting: {scores.get('formatting','N/A')}%, ATS: {scores.get('ats','N/A')}%."
        )

    return (
        "I can help with your resume score, missing skills, ATS-friendliness, or "
        "project suggestions. Try asking: 'Why is my ATS score low?' or "
        "'What skills should I learn?'"
    )


def call_external_api(question: str, context: dict) -> str:
    """Placeholder showing where to plug in OpenAI or Gemini for real
    generative answers. Requires `pip install openai` (or `google-generativeai`)
    and an API key set as an environment variable."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "No external API key configured. Set OPENAI_API_KEY to enable this."

    # Example (uncomment once you've installed the openai package):
    #
    # from openai import OpenAI
    # client = OpenAI(api_key=api_key)
    # prompt = f"Resume analysis context: {context}\n\nUser question: {question}"
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # return response.choices[0].message.content

    return "External API call not implemented yet - see the comment block in chatbot.py."

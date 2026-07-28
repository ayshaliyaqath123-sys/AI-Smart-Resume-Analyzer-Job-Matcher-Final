"""
visualizer.py
Builds chart data for a live skill-gap visualization so the results page can
render an interactive chart with hover tooltips and smooth transitions.
"""

def generate_skill_gap_chart(required_skills: list, found_skills: list) -> dict:
    found_set = set(found_skills)
    labels = required_skills
    values = [1 if skill in found_set else 0 for skill in labels]
    statuses = ["Matched" if value == 1 else "Missing" for value in values]
    colors = ["#22c55e" if value == 1 else "#f59e0b" for value in values]

    return {
        "labels": labels,
        "values": values,
        "statuses": statuses,
        "colors": colors,
        "backgroundColors": [
            "rgba(34, 197, 94, 0.9)" if value == 1 else "rgba(245, 158, 11, 0.92)"
            for value in values
        ],
    }

"""
visualizer.py
Renders a horizontal bar chart comparing skills the resume HAS vs skills
REQUIRED for the target category, and returns it as a base64 PNG string
so it can be embedded directly in an <img> tag with no file saved to disk.
"""
import io
import base64
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for a web server
import matplotlib.pyplot as plt


def generate_skill_gap_chart(required_skills: list, found_skills: list) -> str:
    found_set = set(found_skills)
    labels = required_skills
    values = [1 if skill in found_set else 0 for skill in labels]
    colors = ["#4CAF50" if v == 1 else "#E0E0E0" for v in values]

    fig_height = max(2, 0.5 * len(labels))
    fig, ax = plt.subplots(figsize=(6, fig_height))

    y_pos = range(len(labels))
    ax.barh(y_pos, [1] * len(labels), color=colors)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.set_title("Skill Gap: Have (green) vs Missing (grey)")
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)

    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

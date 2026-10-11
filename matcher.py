"""Text similarity and transparent skill overlap calculations."""

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# A small editable vocabulary for a readable demo. Add domain-specific skills here.
SKILLS = {
    "python": ("python",), "java": ("java",), "c": ("c", "c language"),
    "c++": ("c++",), "javascript": ("javascript", "js"), "typescript": ("typescript",),
    "html": ("html",), "css": ("css",), "sql": ("sql",), "flask": ("flask",),
    "django": ("django",), "react": ("react",), "node.js": ("node.js", "nodejs"),
    "git": ("git",), "github": ("github",), "linux": ("linux",),
    "data structures": ("data structures",), "algorithms": ("algorithms", "algorithm design"),
    "machine learning": ("machine learning", "ml"), "deep learning": ("deep learning",),
    "artificial intelligence": ("artificial intelligence", " ai "), "nlp": ("nlp", "natural language processing"),
    "pandas": ("pandas",), "numpy": ("numpy",), "scikit-learn": ("scikit-learn", "sklearn"),
    "tensorflow": ("tensorflow",), "pytorch": ("pytorch",), "data analysis": ("data analysis",),
    "statistics": ("statistics",), "excel": ("excel",), "communication": ("communication",),
    "leadership": ("leadership",), "project management": ("project management",),
    "problem solving": ("problem solving", "problem-solving"), "cloud computing": ("cloud computing",),
    "aws": ("aws", "amazon web services"), "azure": ("azure",), "docker": ("docker",),
    "kubernetes": ("kubernetes",), "rest apis": ("rest api", "restful api", "rest apis"),
    "agile": ("agile",), "power bi": ("power bi",), "tableau": ("tableau",),
}


def _normalise(text: str) -> str:
    return " " + re.sub(r"\s+", " ", text.lower().replace("\u2011", "-")) + " "


def extract_skills(text: str) -> list[str]:
    normalized = _normalise(text)
    found = []
    for skill, aliases in SKILLS.items():
        for alias in aliases:
            if re.search(r"(?<![\w+#.])" + re.escape(alias.strip()) + r"(?![\w+#.])", normalized):
                found.append(skill)
                break
    return sorted(found)


def analyze_match(resume_text: str, job_text: str) -> dict:
    """Return a 0-100 TF-IDF cosine score and an explainable skill comparison."""
    if not resume_text.strip() or not job_text.strip():
        raise ValueError("Both resume text and job description are required.")
    try:
        matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(
            [resume_text, job_text]
        )
        score = round(float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0]) * 100)
    except ValueError:
        score = 0

    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_text))
    matched = sorted(resume_skills & job_skills)
    missing = sorted(job_skills - resume_skills)
    recommendations = []
    if matched:
        highlights = ", ".join(matched[:4])
        recommendations.append({
            "title": "Bring your strongest overlap forward",
            "body": f"Your resume and this role both mention {highlights}. If these reflect real experience, feature the most relevant examples near the top of your resume and describe what you did with them.",
        })
        recommendations.append({
            "title": "Add evidence to the keywords",
            "body": "For each relevant skill you keep, add a concise example from a project, course, internship, or job. Explain your contribution and the result; include a number only when you can support it.",
        })
    else:
        recommendations.append({
            "title": "Look for relevant evidence",
            "body": "No shared skills were detected from the built-in vocabulary. Review your coursework, projects, and experience for genuinely relevant examples, including skills described with different wording.",
        })

    if missing:
        gap_names = ", ".join(missing[:5])
        suffix = " and other listed skills" if len(missing) > 5 else ""
        recommendations.append({
            "title": "Review these possible gaps",
            "body": f"The job description mentions {gap_names}{suffix}, which the vocabulary did not find in your resume. If you have used any of them, add a specific example. If you have not, do not claim them; consider building relevant experience first.",
        })
    else:
        recommendations.append({
            "title": "Check the requirements in context",
            "body": "No job skills from the built-in vocabulary were flagged as missing. Read the full posting for requirements that this simple checker may not recognize, such as years of experience, tools, or domain knowledge.",
        })

    recommendations.append({
        "title": "Tailor the summary to this role",
        "body": "Use a short opening summary that connects your most relevant experience to the role. Reuse the employer's wording only when it accurately describes your background.",
    })
    return {
        "score": score,
        "matched_skills": matched,
        "missing_skills": missing,
        "resume_skill_count": len(resume_skills),
        "job_skill_count": len(job_skills),
        "has_skill_data": bool(job_skills),
        "recommendations": recommendations,
    }

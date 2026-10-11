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
    return {
        "score": score,
        "matched_skills": matched,
        "missing_skills": missing,
        "resume_skill_count": len(resume_skills),
        "job_skill_count": len(job_skills),
        "has_skill_data": bool(job_skills),
    }

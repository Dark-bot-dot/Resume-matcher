# AI Resume and Job Matcher

A small web app that compares resume text with a job description. It uses TF-IDF and cosine similarity for a text-overlap score, then reports skills found in both texts and skills present in the job description but not detected in the resume.

## Features

- Paste resume and job-description text, or upload a PDF, DOCX, or TXT resume (maximum 5 MB).
- See a 0-100 TF-IDF cosine similarity score.
- See matched and potentially missing skills from an editable vocabulary in `matcher.py`.
- Get targeted next-step suggestions based on the overlap and detected gaps, with reminders to include only accurate experience.
- Resume uploads are parsed in memory and are not written to disk or saved between requests.
- Responsive interface and Docker deployment option.

## Requirements

- Python 3.10 or newer
- pip

## Run locally

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python app.py
```

Open <http://127.0.0.1:5000>.

## Run with Docker

```bash
docker build -t ai-resume-job-matcher .
docker run --rm -p 8000:8000 -e SECRET_KEY="replace-with-a-long-random-secret" ai-resume-job-matcher
```

Open <http://127.0.0.1:8000>. For a production deployment, configure a unique secret key and use HTTPS at the hosting provider.

## How matching works

1. The app fits a scikit-learn `TfidfVectorizer` on the two submitted texts.
2. It calculates cosine similarity between their TF-IDF vectors and displays the result as a percentage.
3. It scans both texts for skill names and aliases in `SKILLS` in `matcher.py`. Skills in both are shown as matched; job skills not found in the resume are shown as potentially missing.

This is a lightweight, rules-based text-matching demo. It does not use a trained language model, understand context or seniority, validate claims, rank candidates, or predict hiring outcomes. Suggestions are generated from detected terms and fixed templates, not from an AI model reviewing the meaning of the resume. The skill vocabulary is deliberately finite and can miss synonyms or skills outside its list. Similarity is not a probability of getting an interview.

## Supported resume files

- PDF: text-based PDFs. Scanned/image-only PDFs need OCR and currently return little or no text.
- DOCX: document paragraphs and tables.
- TXT: UTF-8 text.

Password-protected PDFs are not supported. If extraction fails, paste the resume text instead.

## Privacy and security notes

- The app does not save uploaded resume files. It parses the request bytes in memory.
- Do not log submitted resume or job-description contents. Current logs report file parsing errors without document text.
- For public hosting, set a strong `SECRET_KEY`, use HTTPS, keep dependencies updated, and add a privacy notice appropriate for your deployment and jurisdiction.
- This is a portfolio project and has not been audited for handling sensitive production data.

## Project structure

```text
ai-resume-job-matcher/
├── app.py                 # Flask routes and in-memory file extraction
├── matcher.py             # TF-IDF scoring and skill detection
├── templates/index.html   # Accessible matcher interface
├── static/styles.css      # Responsive styling
├── requirements.txt
├── Dockerfile
├── .env.example
├── .gitignore
└── README.md
```

## License

MIT. See [LICENSE](LICENSE).

import os
import sys
import requests
import base64
from pprint import pprint
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional imports with fallbacks
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    pdfplumber = None

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    Document = None

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    PyPDF2 = None

# ✅ ONLY LLM SOURCE
try:
    from settings import llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    llm = None

from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

GITHUB_ACCESS_TOKEN = os.environ.get('GITHUB_ACCESS_TOKEN')


# ─────────────────────────────────────────────
# GitHub functions
# ─────────────────────────────────────────────

def get_github_repo_languages(username: str, token: str = None) -> list[str]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com/users/{username}/repos?per_page=100"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("Error fetching repos")
        return []

    repos = response.json()
    languages = set()

    for repo in repos:
        lang_url = repo['languages_url']
        res = requests.get(lang_url, headers=headers)
        if res.status_code == 200:
            for lang in res.json().keys():
                languages.add(lang)

    return list(languages)


def get_readme_content(username: str, token: str = None) -> str | None:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com/repos/{username}/{username}/contents/README.md"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return None

    try:
        content = response.json()['content']
        return base64.b64decode(content).decode('utf-8')
    except:
        return None


# ─────────────────────────────────────────────
# LLM-based skill extraction
# ─────────────────────────────────────────────

def get_combined_skills_llm(repo_languages, readme_text):
    if not readme_text:
        readme_text = "No README"

    system_prompt = """You are an expert tech skill extractor.
Extract ALL technical skills including:
- programming languages
- frameworks
- tools
- databases
- technologies

Return ONLY a comma-separated list.
No explanation."""

    user_prompt = f"""
Repo Languages: {repo_languages}
README: {readme_text}
"""

    # ✅ SINGLE LLM CALL
    if LLM_AVAILABLE and llm:
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            text = response.content

            return set(s.strip() for s in text.split(",") if s.strip())
        except Exception as e:
            print("LLM failed:", e)

    # fallback
    return set(repo_languages)


# ─────────────────────────────────────────────
# Resume parsing
# ─────────────────────────────────────────────

def read_resume_file(file_path):
    if file_path.endswith(".pdf"):
        text = ""

        if PDFPLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(file_path) as pdf:
                    for p in pdf.pages:
                        text += p.extract_text() or ""
                return text
            except:
                pass

        if PYPDF2_AVAILABLE:
            try:
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for p in reader.pages:
                        text += p.extract_text() or ""
                return text
            except:
                pass

    elif file_path.endswith(".docx") and DOCX_AVAILABLE:
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)

    elif file_path.endswith(".txt"):
        return open(file_path).read()

    return None


def extract_skills_from_resume(resume_text):
    system_prompt = "Extract technical skills. Return comma-separated only."

    if LLM_AVAILABLE and llm:
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=resume_text)
            ])
            text = response.content
            return set(s.strip() for s in text.split(",") if s.strip())
        except:
            pass

    # fallback
    return set()


# ─────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────

def get_final_skills_data(username=None, resume_file_path=None):
    if resume_file_path:
        text = read_resume_file(resume_file_path)
        if text:
            return extract_skills_from_resume(text)

    elif username:
        langs = get_github_repo_languages(username, GITHUB_ACCESS_TOKEN)
        readme = get_readme_content(username, GITHUB_ACCESS_TOKEN)
        return get_combined_skills_llm(langs, readme)

    return set()


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    skills = get_final_skills_data(username="Satyajeet-Das")

    if skills:
        print("\nSkills:")
        pprint(sorted(list(skills)))
    else:
        print("No skills found")
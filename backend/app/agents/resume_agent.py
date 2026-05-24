# resume_strategist.py
"""
Resume Strategist
- Receives all data from supervisor (no prompting)
- Fetches GitHub profile + repos + READMEs
- Uses LLM to combine README + description into rich project descriptions
- Selects best template based on skills, role, experience
- Picks best 3-4 projects based on target role and impact
- Fills selected .tex template via LLM
- Runs a simple critic loop (max 3 iterations) to improve output
- Saves resume_output/resume.tex
- Exposes:
    class ResumeStrategist
    function run_interactive_mode()   <- standalone use only
"""

import os
import re
import json
import requests
from settings import llm
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import base64

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

load_dotenv()

# Import skill extractor from skills.py — LLM-based, better than keyword matching
try:
    from skills import get_final_skills_data as _get_skills_from_source
    SKILLS_MODULE_AVAILABLE = True
except ImportError:
    SKILLS_MODULE_AVAILABLE = False
    _get_skills_from_source = None

# Import resume parser for resume_path support
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
GROQ_KEY     = os.getenv("GROQ_API_KEY")


def _find_templates_dir() -> Path:
    """Walk up from this file looking for db/templates/."""
    here = Path(__file__).resolve().parent
    for _ in range(4):
        candidate = here / "db" / "templates"
        if candidate.exists():
            return candidate
        here = here.parent
    return Path(__file__).resolve().parent / "templates"


TEMPLATES_DIR = _find_templates_dir()

COMMON_TECH_KEYWORDS = [
    "python", "java", "javascript", "typescript", "react", "node", "express", "django",
    "flask", "go", "rust", "c++", "c#", "php", "ruby", "swift", "kotlin",
    "docker", "kubernetes", "aws", "gcp", "azure", "sql", "postgres", "mysql", "mongodb",
    "redis", "tensorflow", "pytorch", "scikit-learn", "opencv", "html", "css", "sass",
    "tailwind", "bootstrap", "graphql", "rest", "solidity", "ethers", "web3",
    "terraform", "ansible", "jenkins", "github actions", "prometheus", "grafana",
    "spark", "hadoop", "airflow", "dbt", "mlflow", "huggingface", "xgboost"
]

# ─────────────────────────────────────────────────────────────────────────────
# Resume file parser
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_resume_file(resume_path: str) -> str:
    """
    Extract raw text from a PDF, DOCX, or TXT resume file.
    Returns empty string if parsing fails or libraries are unavailable.
    """
    if not resume_path:
        return ""
    path = Path(resume_path)
    if not path.exists():
        print(f"[RESUME] resume_path not found: {resume_path}")
        return ""

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        if not PDF_AVAILABLE:
            print("[RESUME] pdfplumber not installed — cannot parse PDF. Run: pip install pdfplumber")
            return ""
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
            return "\n".join(pages)
        except Exception as e:
            print(f"[RESUME] PDF parse error: {e}")
            return ""

    if suffix in (".docx", ".doc"):
        if not DOCX_AVAILABLE:
            print("[RESUME] python-docx not installed — cannot parse DOCX. Run: pip install python-docx")
            return ""
        try:
            doc = DocxDocument(str(path))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            print(f"[RESUME] DOCX parse error: {e}")
            return ""

    if suffix == ".txt":
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[RESUME] TXT read error: {e}")
            return ""

    print(f"[RESUME] Unsupported resume file type: {suffix}")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# GitHub fetch helpers with README support
# ─────────────────────────────────────────────────────────────────────────────

def fetch_github_profile(username: str, timeout: int = 10) -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    try:
        r = requests.get(
            f"https://api.github.com/users/{username}",
            headers=headers, timeout=timeout
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def fetch_github_repos(username: str, per_page: int = 20, timeout: int = 10) -> List[dict]:
    """Fetch up to 20 repos so project selection has a good pool."""
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    try:
        r = requests.get(
            f"https://api.github.com/users/{username}/repos",
            headers=headers,
            params={"sort": "updated", "per_page": per_page},
            timeout=timeout,
        )
        r.raise_for_status()
        return [
            {
                "name":        repo.get("name", ""),
                "description": repo.get("description") or "",
                "language":    repo.get("language") or "",
                "stars":       repo.get("stargazers_count", 0),
                "forks":       repo.get("forks_count", 0),
                "html_url":    repo.get("html_url", ""),
                "topics":      repo.get("topics", []),
                "owner":       repo.get("owner", {}).get("login", username)
            }
            for repo in r.json()
        ]
    except Exception:
        return []


def fetch_repo_readme(owner: str, repo_name: str, timeout: int = 10) -> str:
    """
    Fetch README content from a GitHub repository.
    Tries README.md first, then falls back to README.
    Returns empty string if README not found or on error.
    """
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    
    # Try common README filenames
    readme_names = ["README.md", "README", "readme.md", "readme"]
    
    for readme_name in readme_names:
        try:
            url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{readme_name}"
            r = requests.get(url, headers=headers, timeout=timeout)
            
            if r.status_code == 200:
                data = r.json()
                # README content is base64 encoded
                content = data.get("content", "")
                if content:
                    decoded = base64.b64decode(content).decode('utf-8', errors='ignore')
                    # Limit README length to avoid token bloat
                    return decoded[:3000]  # First 3000 chars
        except Exception:
            continue
    
    return ""


def fetch_github_data(username: str) -> Dict[str, Any]:
    """
    Single fetch — result reused for both skills extraction and project selection.
    Now also fetches README for each repo.
    """
    if not username:
        return {"profile": {}, "repos": []}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            pf = ex.submit(fetch_github_profile, username)
            rp = ex.submit(fetch_github_repos, username)
            profile = pf.result()
            repos = rp.result()
        
        # Fetch READMEs for repos (limit to top 10 to avoid rate limits)
        print(f"[RESUME] Fetching READMEs for top {min(len(repos), 10)} repositories...")
        repos_with_readme = []
        
        for repo in repos[:10]:  # Limit to top 10 repos
            readme_content = fetch_repo_readme(repo["owner"], repo["name"])
            repo_copy = repo.copy()
            repo_copy["readme"] = readme_content
            repos_with_readme.append(repo_copy)
            
            if readme_content:
                print(f"[RESUME] ✓ README found for {repo['name']} ({len(readme_content)} chars)")
            else:
                print(f"[RESUME] ✗ No README for {repo['name']}")
        
        # Add remaining repos without README fetch
        repos_with_readme.extend(repos[10:])
        
        return {"profile": profile, "repos": repos_with_readme}
    except Exception as e:
        print(f"[RESUME] GitHub data fetch error: {e}")
        return {"profile": {}, "repos": []}


def enhance_project_description(
    repo_name: str,
    short_description: str,
    readme_content: str,
    language: str,
    llm: Any
) -> str:
    """
    Use LLM to combine repo description and README into a concise,
    professional project description suitable for a resume.
    
    Returns enhanced description or falls back to short_description if LLM fails.
    """
    # If no README, just return the short description
    if not readme_content:
        return short_description or f"Developed {repo_name} using {language}."
    
    prompt = f"""You are a resume writing expert. Create a concise, professional project description by combining the repository description and README content below.

Repository Name: {repo_name}
Language: {language}
Short Description: {short_description or "Not provided"}

README Content (excerpt):
{readme_content[:2000]}

Create a 2-3 sentence professional project description that:
1. Explains what the project does
2. Highlights key technical features or achievements
3. Is suitable for a resume
4. Avoids generic phrases like "This is a project that..."
5. Focuses on impact and technical depth

Output ONLY the description, no preamble."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        enhanced = response.content.strip()
        
        # Validate output isn't too long
        if len(enhanced) > 500:
            # Take first 2 sentences
            sentences = re.split(r'[.!?]+', enhanced)
            enhanced = '. '.join(sentences[:2]) + '.'
        
        return enhanced
    except Exception as e:
        print(f"[RESUME] Description enhancement failed for {repo_name}: {e}")
        return short_description or f"Developed {repo_name} using {language}."


# ─────────────────────────────────────────────────────────────────────────────
# Skill extraction from GitHub data
# ─────────────────────────────────────────────────────────────────────────────

def extract_skills_from_github(github_data: Dict[str, Any]) -> List[str]:
    skills = set()
    for repo in github_data.get("repos", []) or []:
        lang = (repo.get("language") or "").strip()
        if lang:
            skills.add(lang)
        text = " ".join([
            repo.get("name", ""),
            repo.get("description", "") or "",
            " ".join(repo.get("topics", [])),
            repo.get("readme", "")[:500]  # Include some README text for skill extraction
        ]).lower()
        for kw in COMMON_TECH_KEYWORDS:
            if kw in text:
                skills.add(kw)
    return sorted({s.strip() for s in skills if s and len(s) < 40}, key=str.lower)


# ─────────────────────────────────────────────────────────────────────────────
# Project selection — best 3-4 for target role WITH README
# ─────────────────────────────────────────────────────────────────────────────

def select_best_projects(
    repos:        List[dict],
    target_role:  str,
    llm:          Any,
    max_projects: int = 4
) -> List[Dict[str, Any]]:
    """
    LLM picks the best repos based on relevance to target role and impact.
    Now includes README-enhanced descriptions.
    Falls back to top-by-stars if LLM call fails.
    """
    if not repos:
        return []

    repo_summary = [
        {
            "index":       i,
            "name":        r["name"],
            "description": r["description"][:150] if r["description"] else "",
            "readme_excerpt": r.get("readme", "")[:200],  # Include README snippet
            "language":    r["language"],
            "stars":       r["stars"],
            "forks":       r["forks"],
            "topics":      r["topics"]
        }
        for i, r in enumerate(repos)
    ]

    prompt_text = (
        "You are a resume expert. Select the {max_projects} most impressive and relevant\n"
        "GitHub repositories to feature on a resume for the target role below.\n\n"
        "Prioritize:\n"
        "1. Relevance to the target role\n"
        "2. Non-empty, meaningful description or README\n"
        "3. Impact signals: stars and forks\n"
        "4. Technical depth and interesting stack\n\n"
        "Target role: {target_role}\n\n"
        "Repositories:\n"
        "{repos}\n\n"
        "Respond ONLY with a JSON array of selected repository indexes (0-based). "
        "Example: [0, 3, 7, 12]\n"
        "No explanation, no markdown."
    )

    try:
        chain = ChatPromptTemplate.from_template(prompt_text) | llm | StrOutputParser()
        raw   = chain.invoke({
            "max_projects": max_projects,
            "target_role":  target_role,
            "repos":        json.dumps(repo_summary, indent=2)
        }).strip()
        raw     = re.sub(r"```[a-z]*", "", raw).replace("```", "").strip()
        indexes = json.loads(raw)

        selected = []
        for idx in indexes[:max_projects]:
            if 0 <= idx < len(repos):
                r = repos[idx]
                
                # Enhance description using README
                enhanced_desc = enhance_project_description(
                    repo_name=r["name"],
                    short_description=r["description"],
                    readme_content=r.get("readme", ""),
                    language=r["language"] or "Multiple",
                    llm=llm
                )
                
                selected.append({
                    "name":                r["name"],
                    "description":         enhanced_desc,  # Enhanced description
                    "description_bullets": [enhanced_desc],
                    "tech_stack":          [r["language"]] if r["language"] else [],
                    "stars":               r["stars"],
                    "html_url":            r["html_url"],
                    "readme":              r.get("readme", "")  # Pass full README for interview
                })
        return selected

    except Exception as e:
        print(f"[RESUME] Project selection LLM call failed: {e}")
        sorted_repos = sorted(repos, key=lambda x: x["stars"], reverse=True)
        
        selected = []
        for r in sorted_repos[:max_projects]:
            # Even in fallback, enhance descriptions
            enhanced_desc = enhance_project_description(
                repo_name=r["name"],
                short_description=r["description"],
                readme_content=r.get("readme", ""),
                language=r["language"] or "Multiple",
                llm=llm
            )
            
            selected.append({
                "name":                r["name"],
                "description":         enhanced_desc,
                "description_bullets": [enhanced_desc],
                "tech_stack":          [r["language"]] if r["language"] else [],
                "stars":               r["stars"],
                "html_url":            r["html_url"],
                "readme":              r.get("readme", "")
            })
        
        return selected


# ─────────────────────────────────────────────────────────────────────────────
# ResumeStrategist
# ─────────────────────────────────────────────────────────────────────────────

class ResumeStrategist:

    def __init__(self):
        if not GROQ_KEY:
            raise Exception("Missing GROQ_API_KEY")
        self.llm = llm

    # ── Template selection ───────────────────────────────────────────────────

    def _select_template(
        self,
        skills:              List[str],
        target_role:         str,
        years_of_experience: float,
        work_experience:     List[dict],
        projects:            List[dict]
    ) -> str:
        """
        Two-axis decision:
          Axis 1 — role type  (skill signals + target_role keywords)
          Axis 2 — seniority  (years_of_experience + work_experience richness)
        Returns a template filename e.g. 'frontend.tex'
        """
        skills_lower = {s.lower() for s in skills}
        role_lower   = target_role.lower()

        ml_signals = {
            "python", "r", "julia",
            "pytorch", "tensorflow", "keras", "jax", "flax",
            "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost",
            "huggingface", "hugging face", "transformers", "diffusers",
            "langchain", "llamaindex", "llama index", "openai", "anthropic",
            "mlflow", "wandb", "weights & biases", "optuna", "ray",
            "nlp", "natural language processing", "computer vision", "cv",
            "deep learning", "machine learning", "reinforcement learning",
            "llm", "large language model", "generative ai", "gen ai",
            "stable diffusion", "bert", "gpt", "t5", "llama",
            "numpy", "pandas", "matplotlib", "seaborn", "plotly",
            "jupyter", "colab", "kaggle",
            "onnx", "torchserve", "triton", "bento ml", "bentoml",
        }

        frontend_signals = {
            "javascript", "typescript", "html", "css",
            "react", "react.js", "reactjs",
            "next.js", "nextjs",
            "vue", "vue.js", "vuejs", "nuxt", "nuxt.js",
            "angular", "angularjs",
            "svelte", "sveltekit",
            "solid", "solidjs",
            "remix", "astro", "gatsby",
            "tailwind", "tailwindcss", "tailwind css",
            "sass", "scss", "less",
            "bootstrap", "material ui", "mui", "chakra ui",
            "styled components", "emotion",
            "webpack", "vite", "parcel", "rollup", "esbuild",
            "babel", "eslint", "prettier",
            "jest", "vitest", "cypress", "playwright", "testing library",
            "redux", "zustand", "mobx", "recoil", "jotai", "pinia",
            "react native", "expo",
        }

        backend_signals = {
            "python", "java", "go", "golang", "rust", "ruby",
            "php", "c#", "c++", "kotlin", "scala", "elixir", "haskell",
            "django", "django rest framework", "drf",
            "flask", "fastapi", "aiohttp", "tornado", "starlette",
            "express", "express.js", "expressjs",
            "nestjs", "nest.js", "koa", "hapi", "fastify",
            "spring", "spring boot", "spring framework",
            "laravel", "symfony", "codeigniter",
            "rails", "ruby on rails", "sinatra",
            "gin", "fiber", "echo", "gorilla",
            "actix", "axum", "rocket",
            ".net", "asp.net", "dotnet",
            "postgresql", "postgres", "mysql", "mariadb",
            "mongodb", "mongoose", "dynamodb", "firestore",
            "redis", "memcached", "cassandra", "couchdb",
            "sqlite", "oracle", "mssql", "sql server",
            "elasticsearch", "opensearch",
            "rest", "restful", "rest api", "graphql", "grpc",
            "websocket", "websockets", "socket.io",
            "microservices", "micro services",
            "rabbitmq", "kafka", "celery", "bull", "sidekiq",
            "sqlalchemy", "prisma", "typeorm", "sequelize", "hibernate",
        }

        devops_signals = {
            "docker", "docker compose", "podman",
            "kubernetes", "k8s", "helm", "kustomize", "openshift",
            "aws", "amazon web services", "ec2", "s3", "lambda",
            "ecs", "eks", "rds", "sqs", "sns", "cloudfront",
            "gcp", "google cloud", "gke", "cloud run", "bigquery",
            "azure", "aks", "azure functions",
            "terraform", "pulumi", "ansible", "chef", "puppet",
            "cloudformation", "cdk",
            "github actions", "gitlab ci", "gitlab-ci", "jenkins",
            "circleci", "travis ci", "argocd", "argo cd",
            "tekton", "drone", "buildkite",
            "prometheus", "grafana", "datadog", "newrelic", "splunk",
            "elk", "elasticsearch", "logstash", "kibana",
            "jaeger", "zipkin", "opentelemetry",
            "nginx", "apache", "caddy", "traefik",
            "vault", "istio", "envoy", "linkerd",
            "bash", "shell", "powershell", "hcl",
        }

        data_signals = {
            "python", "sql", "scala", "r",
            "spark", "apache spark", "pyspark",
            "hadoop", "hive", "hbase", "flink",
            "kafka", "apache kafka",
            "airflow", "apache airflow", "prefect", "dagster", "luigi",
            "dbt", "dbt-core",
            "bigquery", "redshift", "snowflake", "databricks",
            "delta lake", "iceberg", "hudi",
            "athena", "presto", "trino",
            "fivetran", "airbyte", "stitch", "kafka connect",
            "flink", "beam", "apache beam",
            "tableau", "power bi", "looker", "metabase", "superset",
            "parquet", "avro", "orc",
        }

        ml_score       = len(skills_lower & ml_signals)
        frontend_score = len(skills_lower & frontend_signals)
        devops_score   = len(skills_lower & devops_signals)
        data_score     = len(skills_lower & data_signals)
        backend_score  = len(skills_lower & backend_signals)

        if any(w in role_lower for w in [
            "machine learning", "ml engineer", "ai engineer", "deep learning",
            "nlp engineer", "computer vision", "data scientist", "research engineer",
            "llm", "generative ai", "gen ai", "ai/ml", "ml/ai"
        ]):
            ml_score += 3

        if any(w in role_lower for w in [
            "frontend", "front-end", "front end", "ui engineer", "ui developer",
            "react developer", "react engineer", "vue developer", "angular developer",
            "next.js", "javascript developer", "typescript developer", "web developer"
        ]):
            frontend_score += 3

        if any(w in role_lower for w in [
            "devops", "dev ops", "sre", "site reliability", "platform engineer",
            "cloud engineer", "infrastructure engineer", "kubernetes engineer",
            "aws engineer", "gcp engineer", "azure engineer", "devsecops"
        ]):
            devops_score += 3

        if any(w in role_lower for w in [
            "data engineer", "analytics engineer", "etl developer",
            "data pipeline", "spark engineer", "databricks", "snowflake"
        ]):
            data_score += 3

        if any(w in role_lower for w in [
            "backend", "back-end", "back end", "server side",
            "python developer", "python engineer", "django developer",
            "node.js developer", "node developer", "java developer",
            "go developer", "golang developer", "api developer",
            "software engineer", "software developer", "full stack", "fullstack"
        ]):
            backend_score += 2

        role_scores = {
            "data_scientist": ml_score + data_score,
            "frontend":       frontend_score,
            "devops":         devops_score,
            "backend":        backend_score,
        }

        best_role  = max(role_scores, key=role_scores.get)
        best_score = role_scores[best_role]

        if best_score < 2:
            if not work_experience or years_of_experience <= 2:
                return "fresher.tex"
            elif years_of_experience <= 6:
                return "mid_level.tex"
            else:
                return "senior_engineer.tex"

        if years_of_experience >= 6 and work_experience:
            return "senior_engineer.tex"

        return f"{best_role}.tex"

    # ── Template loader ──────────────────────────────────────────────────────

    def _load_template(self, filename: str) -> str:
        path = TEMPLATES_DIR / filename
        if not path.exists():
            fallback = Path(__file__).resolve().parent / "templates" / filename
            if fallback.exists():
                path = fallback
            else:
                raise FileNotFoundError(f"Template not found: {filename}")
        return path.read_text(encoding="utf-8")

    # ── LLM fills the template ───────────────────────────────────────────────

    def _fill_template(
        self,
        template_tex: str,
        data:         Dict[str, Any],
        feedback:     str = ""
    ) -> str:
        """
        LLM fills the .tex template with real candidate data.
        Uses direct HumanMessage construction so LaTeX braces in template_tex
        are never misread as prompt variables.
        """
        feedback_section = (
            f"\nCRITIC FEEDBACK TO ADDRESS IN THIS REVISION:\n{feedback}\n"
            if feedback else ""
        )

        prompt = (
            "You are an expert LaTeX resume builder.\n\n"
            "Fill the given .tex resume template with the candidate's real data below.\n\n"
            "Rules:\n"
            r"1. Replace all \def\Var... variables at the top with real values." + "\n"
            r"2. Replace placeholder \workentry, \projectentry, \eduentry commands with real data." + "\n"
            r"3. REMOVE any section (including its \section{...} heading) that has no data." + "\n"
            "   Example: if work_experience is empty, delete the entire Work Experience section.\n"
            "4. Write strong, concise bullet points. Use metrics and numbers wherever the data allows.\n"
            "5. For senior roles: emphasise leadership, team scope, architecture decisions.\n"
            "6. For fresher/no-experience profiles: emphasise technical depth of projects.\n"
            "7. Output ONLY the complete compile-ready LaTeX. No explanation, no markdown fences.\n"
            + feedback_section + "\n"
            "TEMPLATE:\n"
            + template_tex + "\n\n"
            "CANDIDATE DATA:\n"
            + json.dumps(data, indent=2)
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()
        except Exception as e:
            print(f"[RESUME] LLM call failed: {e}")
            return template_tex

        # Strip opening fence plus any language tag
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\n?```$", "", raw.strip())

        return raw.strip()

    # ── Critic ───────────────────────────────────────────────────────────────

    def _critique(self, latex: str, data: Dict[str, Any]) -> Tuple[int, str]:
        """
        Scores the resume 1-10 and returns specific issues to fix.
        """
        target_role = data.get("target_role", "Software Engineer")
        prompt_str = (
            "You are a senior technical recruiter reviewing a LaTeX resume.\n\n"
            f"Score this resume 1-10 for the target role: {target_role}\n\n"
            "Check:\n"
            '- No placeholder text remaining (e.g. "Full Name", "Company Name", "University Name")\n'
            "- Bullet points are specific and quantified, not generic\n"
            "- Empty sections have been removed\n"
            "- Summary matches the target role\n"
            "- Overall impression: would you shortlist this candidate?\n\n"
            "LaTeX resume (first 5000 chars):\n"
            + latex[:5000]
            + '\n\nRespond ONLY with JSON: {"score": <int 1-10>, "issues": "<specific problems to fix>"}\n'
            "No markdown, no extra text."
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt_str)])
            raw = response.content if hasattr(response, "content") else str(response)
            raw = re.sub(r"```[a-z]*", "", raw).replace("```", "").strip()
            result = json.loads(raw)
            return int(result.get("score", 5)), str(result.get("issues", ""))
        except Exception:
            return 5, "Could not parse critic response."

    # ── Project suggestions fallback ─────────────────────────────────────────

    def _generate_project_suggestions(
        self,
        target_role: str,
        skills:      List[str]
    ) -> List[Dict[str, Any]]:
        """
        Uses LLM to suggest 2-3 project ideas based on target role and existing skills.
        Returns a list of project dicts with name, description_bullets, tech_stack.
        Falls back to empty list if LLM call fails.
        """
        prompt_str = (
            "You are a senior software engineer mentoring a junior developer.\n\n"
            f"Target Role: {target_role}\n"
            f"Candidate's Current Skills: {', '.join(skills[:20]) if skills else 'general programming'}\n\n"
            "Suggest 2-3 portfolio project ideas this candidate could build to strengthen their resume "
            "for the target role. Each project should be achievable in 2-4 weeks and highly relevant.\n\n"
            "Respond ONLY with a JSON array. Each element must have:\n"
            '  "name": short project name\n'
            '  "description_bullets": list of 2 bullet points describing what it does / impact\n'
            '  "tech_stack": list of 3-5 relevant technologies\n\n'
            "No markdown, no explanation, just the JSON array."
        )
        try:
            response = self.llm.invoke([HumanMessage(content=prompt_str)])
            raw = response.content if hasattr(response, "content") else str(response)
            raw = re.sub(r"```[a-z]*", "", raw).replace("```", "").strip()
            suggestions = json.loads(raw)
            if not isinstance(suggestions, list):
                return []
            return suggestions[:3]
        except Exception as e:
            print(f"[RESUME] Project suggestion LLM call failed: {e}")
            return []

    # ── Inventory ────────────────────────────────────────────────────────────

    def _make_inventory(self, data: Dict[str, Any], template_used: str) -> str:
        contact  = data.get("contact", {})
        skills   = data.get("skills", [])[:20]
        projects = data.get("projects", [])[:4]

        lines = [
            "",
            "RESUME INVENTORY SUMMARY",
            "------------------------",
            f"Name:          {contact.get('name', 'N/A')}",
            f"Target Role:   {data.get('target_role', 'N/A')}",
            f"Template Used: {template_used}",
            f"Top Skills:    {', '.join(skills)}",
            "",
            "Selected Projects:"
        ]
        for p in projects:
            lines.append(f"  - {p.get('name', '')}")
            desc = p.get("description", "")
            if desc:
                lines.append(f"      {desc[:150]}...")
        return "\n".join(lines)

    # ── Master method ────────────────────────────────────────────────────────

    def generate_resume(
        self,
        target_role:         str,
        github_username:     Optional[str],
        resume_path:         Optional[str],
        user_skills:         List[str],
        contact:             Optional[Dict[str, Any]] = None,
        education:           Optional[List[Dict[str, Any]]] = None,
        work_experience:     Optional[List[Dict[str, Any]]] = None,
        years_of_experience: float = 0.0
    ) -> Dict[str, Any]:
        """
        Called by supervisor_agent. No user prompting.
        Returns: {resume_json, resume_inventory, latex_path, template_used, projects}
        """

        projects      = []
        merged_skills = list(user_skills or [])

        # ── Parse resume file for additional skills if provided ──────────────
        if resume_path:
            print(f"[RESUME] Parsing resume file: {resume_path}")
            resume_text = extract_text_from_resume_file(resume_path)
            if resume_text and SKILLS_MODULE_AVAILABLE and _get_skills_from_source is not None:
                print("[RESUME] Extracting skills from resume file via skills.py")
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                                     delete=False, encoding='utf-8') as tmp:
                        tmp.write(resume_text)
                        tmp_path = tmp.name
                    file_skills = list(_get_skills_from_source(None, tmp_path))
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                    before = len(merged_skills)
                    merged_skills = list(dict.fromkeys([*merged_skills, *file_skills]))
                    print(f"[RESUME] Resume file added {len(merged_skills) - before} new skills")
                except Exception as e:
                    print(f"[RESUME] Skill extraction from file failed: {e}")

        # ── Single GitHub fetch with README support ──────────────────────────
        if github_username:
            print(f"[RESUME] Fetching GitHub: {github_username}")

            github_data = fetch_github_data(github_username)
            repos       = github_data.get("repos", [])
            profile     = github_data.get("profile", {}) or {}

            # Skill extraction
            if SKILLS_MODULE_AVAILABLE and _get_skills_from_source is not None:
                print("[RESUME] Using skills.py for GitHub skill extraction")
                try:
                    github_skills = list(_get_skills_from_source(github_username, None))
                except Exception as e:
                    print(f"[RESUME] skills.py extraction failed: {e} — using keyword fallback")
                    github_skills = extract_skills_from_github(github_data)
            else:
                print("[RESUME] skills.py not available — using keyword fallback")
                github_skills = extract_skills_from_github(github_data)

            # Merge skills
            merged_skills = list(dict.fromkeys([*merged_skills, *github_skills]))
            print(f"[RESUME] Total skills after GitHub merge: {len(merged_skills)}")

            # Project selection with README-enhanced descriptions
            projects = select_best_projects(repos, target_role, self.llm, max_projects=4)

            # Project suggestions if not enough found
            if not projects or len(projects) < 2:
                print("\n[RESUME] Not enough strong projects found on GitHub.\n")
                suggested = self._generate_project_suggestions(
                    target_role=target_role,
                    skills=merged_skills
                )
                if suggested:
                    print("Project ideas to build (not added to resume — for your reference):\n")
                    for i, p in enumerate(suggested, 1):
                        print(f"  {i}. {p.get('name', '')}")
                        for b in p.get("description_bullets", [])[:2]:
                            print(f"     - {b}")
                        print(f"     Tech: {', '.join(p.get('tech_stack', []))}\n")

            print(f"[RESUME] Selected {len(projects)} projects from {len(repos)} repos")

            # Fill contact gaps from GitHub profile
            if contact is None:
                contact = {}
            if not contact.get("name") and profile.get("name"):
                contact["name"] = profile["name"]
            if not contact.get("location") and profile.get("location"):
                contact["location"] = profile["location"]

        # ── Build data payload ───────────────────────────────────────────────
        data = {
            "target_role":         target_role or "Software Engineer",
            "contact":             contact or {},
            "education":           education or [],
            "work_experience":     work_experience or [],
            "skills":              merged_skills,
            "projects":            projects,
            "years_of_experience": years_of_experience
        }

        # ── Select template ──────────────────────────────────────────────────
        template_filename = self._select_template(
            skills              = merged_skills,
            target_role         = target_role,
            years_of_experience = years_of_experience,
            work_experience     = work_experience or [],
            projects            = projects
        )
        print(f"[RESUME] Template selected: {template_filename}")
        template_tex = self._load_template(template_filename)

        # ── Fill + critic loop ───────────────────────────────────────────────
        MAX_ITERATIONS  = 3
        SCORE_THRESHOLD = 7
        latex    = ""
        feedback = ""
        score    = 0

        for iteration in range(1, MAX_ITERATIONS + 1):
            print(f"[RESUME] Iteration {iteration}/{MAX_ITERATIONS}")
            latex = self._fill_template(template_tex, data, feedback=feedback)
            score, feedback = self._critique(latex, data)
            print(f"[RESUME] Critic score: {score}/10")

            if score >= SCORE_THRESHOLD:
                print(f"[RESUME] Accepted at iteration {iteration}")
                break
            if iteration < MAX_ITERATIONS:
                print(f"[RESUME] Revising — issues: {feedback}")

        # ── Save ─────────────────────────────────────────────────────────────
        tex_path = ""
        try:
            os.makedirs("resume_output", exist_ok=True)
            tex_path = os.path.join("resume_output", "resume.tex")
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(latex)
            print(f"[RESUME] Saved to {tex_path} (score: {score}/10)")
        except Exception as e:
            print(f"[RESUME] Could not save: {e}")

        # ── Return ───────────────────────────────────────────────────────────
        resume_json = {
            "contact":         data["contact"],
            "summary":         f"Resume targeting {target_role}",
            "skills":          merged_skills,
            "education":       data["education"],
            "work_experience": data["work_experience"],
            "projects":        projects,
            "template_used":   template_filename,
            "critic_score":    score
        }

        out = {
            "resume_json":      resume_json,
            "resume_inventory": self._make_inventory(data, template_filename),
            "template_used":    template_filename,
            "projects":         projects,  # Contains enhanced descriptions + README
            "latex_path":       tex_path,
        }
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Standalone interactive mode
# ─────────────────────────────────────────────────────────────────────────────

def run_interactive_mode() -> Dict[str, Any]:
    print("\n===============================")
    print("   INTERACTIVE RESUME BUILDER")
    print("===============================\n")

    target_role = input("Target job role: ").strip()
    gh          = input("GitHub username (ENTER to skip): ").strip()
    try:
        years = int(input("Years of experience: ").strip())
    except ValueError:
        years = 0

    github_data = fetch_github_data(gh) if gh else {}
    profile     = github_data.get("profile", {}) if github_data else {}
    gh_name     = profile.get("name", "")

    name     = input(f"Full name [{gh_name}]: ").strip() or gh_name
    email    = input("Email: ").strip()
    phone    = input("Phone: ").strip()
    location = input("Location: ").strip() or profile.get("location", "")

    contact = {"name": name, "email": email, "phone": phone,
               "location": location, "github": gh}

    education = []
    print("\n--- Education (blank Institution to stop) ---")
    while True:
        inst = input("  Institution: ").strip()
        if not inst:
            break
        education.append({
            "institution": inst,
            "degree":      input("  Degree: ").strip(),
            "field":       input("  Field: ").strip(),
            "status":      input("  Status: ").strip()
        })

    work_experience = []
    print("\n--- Work Experience (blank Company to stop) ---")
    while True:
        comp = input("  Company: ").strip()
        if not comp:
            break
        bullets = []
        print("  Bullets ('done' to stop):")
        while len(bullets) < 4:
            b = input("    - ").strip()
            if not b or b.lower() == "done":
                break
            bullets.append(b)
        work_experience.append({
            "company":             comp,
            "position":            input("  Position: ").strip(),
            "duration":            input("  Duration: ").strip(),
            "description_bullets": bullets
        })

    raw_skills  = input("\nYour skills (comma-separated, ENTER to skip): ").strip()
    user_skills = [s.strip() for s in raw_skills.split(",") if s.strip()] if raw_skills else []

    strategist = ResumeStrategist()
    result = strategist.generate_resume(
        target_role         = target_role,
        github_username     = gh or None,
        resume_path         = None,
        user_skills         = user_skills,
        contact             = contact,
        education           = education,
        work_experience     = work_experience,
        years_of_experience = years
    )

    print(f"\n  Template : {result.get('template_used')}")
    print(f"  Saved at : {result.get('latex_path') or 'N/A'}")
    print(result.get("resume_inventory", ""))
    return result


if __name__ == "__main__":
    run_interactive_mode()
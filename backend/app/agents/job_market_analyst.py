# job_market_analyst.py
"""
Job Market Analyst Agent
- Fetches live jobs via JSearch API (with spaCy + LLM skill enrichment)
- Stores new jobs in Chroma vector DB for future use
- Falls back to vector DB if JSearch is unavailable or rate-limited
- Performs skill gap analysis against user's actual skills
- Compiles LangGraph workflow at module level so supervisor can call app.invoke()
- PostgreSQL checkpointer used if DB_URL available, otherwise runs without persistence
"""

import os
import re
import json
import time
import requests
from datetime import datetime, timedelta
import spacy
from spacy.matcher import PhraseMatcher
from typing import Dict, Any, Literal, Annotated, List
from collections import Counter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FIX #2: corrected import — langchain_classic does not exist
# ✅ NEW (correct for newer langchain)
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# ─────────────────────────────────────────────────────────────────────────────
# Settings import
# ─────────────────────────────────────────────────────────────────────────────

print("[JMA] Loading settings...")
from settings import (
    llm,
    embedding_function,
    CHROMA_PERSIST_DIR,
    DB_URL,
    COHERE_API_KEY
)

# ─────────────────────────────────────────────────────────────────────────────
# JSearch config
# ─────────────────────────────────────────────────────────────────────────────

JSEARCH_API_KEY  = os.getenv("JSEARCH_API_KEY", "")
JSEARCH_BASE_URL = "https://jsearch.p.rapidapi.com"
JSEARCH_HEADERS  = {
    "X-RapidAPI-Key":  JSEARCH_API_KEY,
    "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
}

# Minimum number of recent cached docs before we skip live fetch
MIN_DB_THRESHOLD = 5

# ─────────────────────────────────────────────────────────────────────────────
# spaCy setup
# ─────────────────────────────────────────────────────────────────────────────

SKILLS = [
    "Python", "JavaScript", "TypeScript", "Java", "Go", "Golang", "Rust",
    "C", "C++", "C#", "PHP", "Ruby", "Swift", "Kotlin", "Scala", "R",
    "Julia", "Perl", "Haskell", "Elixir", "Erlang", "Clojure", "Lua",
    "Bash", "Shell", "PowerShell", "MATLAB",
    "HTML", "HTML5", "CSS", "CSS3", "SASS", "SCSS", "Less",
    "React", "React.js", "Next.js", "Vue", "Vue.js", "Nuxt.js",
    "Angular", "AngularJS", "Svelte", "SvelteKit", "Remix", "Astro", "Gatsby",
    "Redux", "Zustand", "MobX", "Recoil", "Pinia",
    "Tailwind", "TailwindCSS", "Bootstrap", "Material UI", "Chakra UI",
    "Styled Components", "Emotion",
    "Webpack", "Vite", "Parcel", "Rollup", "Babel", "ESLint",
    "Jest", "Vitest", "Cypress", "Playwright", "Testing Library",
    "React Native", "Expo",
    "Django", "Flask", "FastAPI", "Starlette", "Tornado", "aiohttp",
    "Express", "Express.js", "NestJS", "Fastify", "Koa", "Hapi",
    "Spring", "Spring Boot", "Spring Framework",
    "Laravel", "Symfony", "CodeIgniter",
    "Ruby on Rails", "Sinatra", "Gin", "Fiber", "Echo", "Gorilla",
    "Actix", "Axum", "Rocket", "ASP.NET", ".NET", "dotnet",
    "GraphQL", "REST", "RESTful", "gRPC", "WebSocket", "Socket.IO",
    "Microservices", "Serverless",
    "PostgreSQL", "MySQL", "MariaDB", "SQLite", "Oracle", "SQL Server",
    "MongoDB", "Mongoose", "DynamoDB", "Firestore", "CouchDB",
    "Redis", "Memcached", "Cassandra", "Elasticsearch", "OpenSearch",
    "InfluxDB", "TimescaleDB", "Neo4j",
    "Prisma", "SQLAlchemy", "TypeORM", "Sequelize", "Hibernate", "SQL", "NoSQL",
    "AWS", "Amazon Web Services", "EC2", "S3", "Lambda", "ECS", "EKS",
    "RDS", "SQS", "SNS", "CloudFront", "Route53", "IAM",
    "GCP", "Google Cloud", "GKE", "Cloud Run", "Cloud Functions",
    "Azure", "AKS", "Azure Functions", "Azure DevOps",
    "Docker", "Docker Compose", "Podman",
    "Kubernetes", "K8s", "Helm", "Kustomize", "OpenShift",
    "Terraform", "Pulumi", "Ansible", "Chef", "Puppet", "CloudFormation",
    "GitHub Actions", "GitLab CI", "Jenkins", "CircleCI", "Travis CI",
    "ArgoCD", "Tekton", "Drone",
    "Prometheus", "Grafana", "Datadog", "New Relic", "Splunk",
    "ELK Stack", "Logstash", "Kibana", "Jaeger", "OpenTelemetry",
    "Nginx", "Apache", "Traefik", "Caddy", "Vault", "Istio", "Envoy", "Linkerd",
    "CI/CD", "DevOps", "SRE",
    "PyTorch", "TensorFlow", "Keras", "JAX", "Flax",
    "scikit-learn", "XGBoost", "LightGBM", "CatBoost",
    "HuggingFace", "Transformers", "Diffusers",
    "LangChain", "LlamaIndex", "OpenAI", "Anthropic",
    "MLflow", "Weights & Biases", "Optuna", "Ray",
    "ONNX", "TorchServe", "Triton", "BentoML",
    "NLP", "Computer Vision", "Deep Learning", "Machine Learning",
    "Reinforcement Learning", "LLM", "Generative AI", "BERT", "GPT", "Stable Diffusion",
    "Apache Spark", "PySpark", "Hadoop", "Hive", "Flink",
    "Apache Kafka", "Kafka", "RabbitMQ", "Celery",
    "Apache Airflow", "Prefect", "Dagster", "Luigi",
    "dbt", "Fivetran", "Airbyte",
    "Snowflake", "BigQuery", "Redshift", "Databricks", "Delta Lake",
    "Athena", "Presto", "Trino",
    "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly",
    "Tableau", "Power BI", "Looker", "Metabase", "Superset", "Jupyter", "Spark SQL",
    "Git", "GitHub", "GitLab", "Bitbucket",
    "Jira", "Confluence", "Notion", "Linear",
    "Postman", "Swagger", "OpenAPI",
    "Agile", "Scrum", "Kanban", "TDD", "BDD", "Unit Testing", "Integration Testing",
    "Linux", "Unix", "Figma", "Storybook",
]

# FIX #7: Skills that need exact/special handling to avoid normalization collision
_SPECIAL_SKILL_MAP = {
    "c++": "C++",
    "c#": "C#",
    ".net": ".NET",
    "c":   "C",
}

_nlp      = spacy.load("en_core_web_sm")
_matcher  = PhraseMatcher(_nlp.vocab, attr="LOWER")
_patterns = [_nlp.make_doc(s.lower()) for s in SKILLS]
_matcher.add("SKILLS", _patterns)


def _extract_skills_spacy(text: str) -> list[str]:
    """Extract known skills from free text using spaCy PhraseMatcher."""
    doc     = _nlp(text.lower())
    matches = _matcher(doc)
    seen, result = set(), []
    for _, start, end in matches:
        key = doc[start:end].text.lower()
        if key not in seen:
            seen.add(key)
            canonical = next((s for s in SKILLS if s.lower() == key), key.title())
            result.append(canonical)
    return result


def _llm_enrich_skills(role: str, description: str, existing_skills: list[str]) -> list[str]:
    """
    Use the LLM to infer additional skills not caught by spaCy.
    Merges results into existing_skills — no separate list.
    Falls back gracefully if LLM call fails.
    """
    prompt = f"""You are a technical recruiter analyzing a job posting.

Job Role: {role}
Job Description (excerpt): {description[:1500]}
Skills already extracted: {", ".join(existing_skills) if existing_skills else "none"}

Your task: identify any additional technical skills, tools, or technologies mentioned
or strongly implied in this job description that are NOT already in the extracted list.

Return ONLY a JSON array of skill name strings. No explanation, no markdown, no extra text.
Example: ["Docker", "REST APIs", "System Design"]
If no additional skills found, return an empty array: []"""

    try:
        response = llm.invoke(prompt)
        raw      = response.content if hasattr(response, "content") else str(response)
        raw      = re.sub(r"```(?:json)?|```", "", raw).strip()
        bonus    = json.loads(raw)
        if not isinstance(bonus, list):
            return existing_skills

        # Merge, deduplicated
        seen   = {s.lower() for s in existing_skills}
        merged = list(existing_skills)
        for skill in bonus:
            if isinstance(skill, str) and skill.lower() not in seen:
                seen.add(skill.lower())
                merged.append(skill)
        return merged

    except Exception as e:
        print(f"[JMA] LLM skill enrichment failed: {e} — using spaCy results only.")
        return existing_skills


def _fetch_jobs_jsearch(role: str, location: str, is_fresher: bool = False) -> list[dict]:
    """
    Fetch jobs from JSearch API directly.
    If is_fresher=True, fetches both FULLTIME and INTERN listings.
    Returns normalized job dicts ready for the analyst.
    """
    if not JSEARCH_API_KEY:
        raise ValueError(
            "JSEARCH_API_KEY env var is not set. "
            "Get a free key at https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch"
        )

    emp_types = ["FULLTIME", "INTERN"] if is_fresher else ["FULLTIME"]
    raw_jobs  = []

    for emp_type in emp_types:
        try:
            resp = requests.get(
                f"{JSEARCH_BASE_URL}/search",
                headers=JSEARCH_HEADERS,
                params={
                    "query":            f"{role} in {location}",
                    "num_pages":        1,
                    "country": "in",
                    "date_posted":      "all",
                    "employment_types": emp_type,
                },
                timeout=100,
            )
            if resp.status_code == 429:
                print(f"[JMA] JSearch rate limit hit for {emp_type} — skipping.")
                continue
            resp.raise_for_status()
            raw_jobs.extend(resp.json().get("data", []))
        except Exception as e:
            print(f"[JMA] JSearch fetch failed for {emp_type}: {e}")

    results = []
    for job in raw_jobs:
        description     = job.get("job_description", "")
        skills_api      = job.get("job_required_skills") or []
        skills_spacy    = _extract_skills_spacy(description)

        # Merge API skills + spaCy, deduplicated
        seen   = {s.lower() for s in skills_api}
        merged = list(skills_api)
        for s in skills_spacy:
            if s.lower() not in seen:
                seen.add(s.lower())
                merged.append(s)

        # LLM enrichment — also handles the empty skills case
        if not merged:
            print(f"[JMA] No skills extracted for '{job.get('job_title', '?')}' — using LLM fallback.")
        final_skills = _llm_enrich_skills(
            role        = job.get("job_title", role),
            description = description,
            existing_skills = merged,
        )

        results.append({
            "title":           job.get("job_title", "N/A"),
            "company":         job.get("employer_name", "N/A"),
            "link":            job.get("job_apply_link") or job.get("job_google_link", "N/A"),
            "location":        (job.get("job_city", "") + ", " + job.get("job_country", "")).strip(", ") or "N/A",
            "list_date":       job.get("job_posted_at_datetime_utc", "N/A"),
            "skills_required": final_skills,
            "seniority":       job.get("job_seniority_level") or "N/A",
            "employment_type": job.get("job_employment_type") or "N/A",
            "industries":      job.get("job_naics_name") or "N/A",
            "salary_min":      job.get("job_min_salary"),
            "salary_max":      job.get("job_max_salary"),
            "is_remote":       job.get("job_is_remote", False),
        })

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Vector store + retriever — wrapped in try/except so import never crashes
# ─────────────────────────────────────────────────────────────────────────────

db                    = None
compression_retriever = None
text_splitter         = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150, length_function=len
)

try:
    def _get_vectorstore() -> Chroma:
        if not os.path.exists(CHROMA_PERSIST_DIR):
            print(f"[JMA] Creating new vector store at {CHROMA_PERSIST_DIR}...")
            return Chroma.from_documents(
                documents=[Document(
                    page_content="Job market database initialised.",
                    metadata={"source": "system_init"}
                )],
                embedding=embedding_function,
                persist_directory=CHROMA_PERSIST_DIR
            )
        print(f"[JMA] Loading vector store from {CHROMA_PERSIST_DIR}...")
        return Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embedding_function
        )

    db             = _get_vectorstore()
    base_retriever = db.as_retriever(search_kwargs={"k": 15})
    reranker       = CohereRerank(
        cohere_api_key=COHERE_API_KEY,
        model="rerank-english-v3.0",
        top_n=5
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )
    print("[JMA] Vector store and retriever ready.")

except Exception as e:
    print(f"[JMA] WARNING: Could not initialise vector store: {e}")
    print("[JMA] Will operate in scrape-only mode (no DB caching).")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_job_as_document(job: Dict[str, Any]) -> Document:
    content = f"""
    Job Title: {job.get('title', 'N/A')}
    Company: {job.get('company', 'N/A')}
    Location: {job.get('location', 'N/A')}
    Seniority Level: {job.get('seniority', 'N/A')}
    Employment Type: {job.get('employment_type', 'N/A')}
    Industries: {job.get('industries', 'N/A')}
    Skills Required: {', '.join(job.get('skills_required', []))}
    """.strip()

    return Document(
        page_content=content,
        metadata={
            "source":     job.get('link', 'N/A'),
            "company":    job.get('company', 'N/A'),
            "location":   job.get('location', 'N/A'),
            "seniority":  job.get('seniority', 'N/A'),
            "title":      job.get('title', 'N/A'),
            "fetched_at": datetime.utcnow().timestamp(),   #
        }
    )


def _normalize_skill(skill: str) -> str:
    """
    FIX #7: Preserve special skills (C++, C#, .NET) before generic normalization
    so they don't all collapse to the same string.
    """
    if not skill:
        return ""
    lower = skill.lower().strip('.,;-')
    # Check special cases first
    if lower in _SPECIAL_SKILL_MAP:
        return _SPECIAL_SKILL_MAP[lower]
    return re.sub(r'[\s-]+', ' ', lower)


def _is_recent_job(doc, days: int = 3) -> bool:
    try:
        ts = doc.metadata.get("fetched_at")
        if not ts:
            return False
        fetched_time = datetime.utcfromtimestamp(ts)
        return datetime.utcnow() - fetched_time < timedelta(days=days)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:            Annotated[list, add_messages]
    role:                str
    location:            str
    user_skills:         List[str]
    years_of_experience: float
    jobs_found:          bool
    scraped:             bool
    analysis_done:       bool
    job_data:            list
    skill_gap_data:      dict
    next_step:           str


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────

def fetch_jobs(state: AgentState) -> AgentState:
    """
    1. Check ChromaDB for recent cached jobs (raw metadata filter — no rerank cost).
    2. If enough recent results exist, return them immediately.
    3. Otherwise call JSearch API, store results, and return.
    4. Falls back to vector DB search if JSearch fails.

    FIX #3: Cache check now uses raw db.get() to avoid paying Cohere rerank on cache hits.
    FIX #4: 'scraped' is always set explicitly on every return path.
    FIX #5: Indentation corrected — cache-hit return is properly scoped.
    """
    print(f"[NODE] Fetching jobs via JSearch: {state['role']} in {state['location']}")

    # ── FIX #3: Cheap cache check via raw Chroma metadata, no reranker cost ──
    if db is not None:
        try:
            cutoff = (datetime.utcnow() - timedelta(days=3)).timestamp()
            # Chroma where-filter: only docs fetched within the last 3 days
            cached_results = db.get(
                where={"fetched_at": {"$gt": cutoff}},
                limit=50,
            )
            cached_docs = cached_results.get("documents", [])
            cached_meta = cached_results.get("metadatas", [])

            # Location-aware filter on cached results
            target_loc = state['location'].lower()
            recent_jobs = []
            for content, meta in zip(cached_docs, cached_meta):
                doc_loc = meta.get("location", "").lower()
                if target_loc in doc_loc or doc_loc in target_loc:
                    skills = []
                    if "skills required:" in content.lower():
                        skills_line = content.lower().split("skills required:")[-1].strip()
                        skills = [s.strip() for s in skills_line.split(",") if s.strip()]
                    recent_jobs.append({
                        "title":           meta.get("title", "N/A"),
                        "company":         meta.get("company", "N/A"),
                        "location":        meta.get("location", "N/A"),
                        "skills_required": skills,
                        "link":            meta.get("source", "N/A"),
                    })

            if len(recent_jobs) >= MIN_DB_THRESHOLD:
                print(f"[NODE] Using {len(recent_jobs)} recent cached jobs (< 3 days old).")
                # FIX #4 & #5: scraped is explicitly set, return is properly indented
                return {
                    **state,
                    "scraped":    True,
                    "jobs_found": True,
                    "job_data":   recent_jobs,
                    "next_step":  "analyze_skills",
                }
        except Exception as e:
            print(f"[NODE] Cache check failed: {e} — proceeding to live fetch.")

    # ── Live fetch via JSearch ──
    all_new_chunks: list = []

    try:
        jobs = _fetch_jobs_jsearch(
            role       = state['role'],
            location   = state['location'],
            is_fresher = state.get('years_of_experience', 1) < 1,
        )

        if not jobs:
            print("[NODE] JSearch returned 0 results — falling back to vector DB.")
            return {**state, "scraped": True, "next_step": "search_db"}

        # Store in vector DB if available
        if db is not None:
            for job in jobs:
                doc    = _format_job_as_document(job)
                chunks = text_splitter.split_documents([doc])
                all_new_chunks.extend(chunks)
            if all_new_chunks:
                print(f"[NODE] Storing {len(all_new_chunks)} chunks in vector DB...")
                db.add_documents(all_new_chunks)

        print(f"[NODE] Fetched {len(jobs)} jobs successfully.")
        return {
            **state,
            "scraped":    True,
            "jobs_found": True,
            "job_data":   jobs,
            "next_step":  "analyze_skills",
        }

    except Exception as e:
        print(f"[NODE] JSearch fetch failed: {e} — falling back to vector DB.")
        return {**state, "scraped": True, "next_step": "search_db"}


def search_db(state: AgentState) -> AgentState:
    """
    Fallback: search the Chroma vector DB for cached job postings.
    Used when JSearch is unavailable or returned empty results.
    Location filter kept as-is per user preference.
    """
    print(f"[NODE] Searching DB: {state['role']} in {state['location']}")

    if compression_retriever is None:
        print("[NODE] Vector DB not available — no data sources left.")
        return {
            **state,
            "jobs_found": False,
            "job_data":   [],
            "next_step":  "analyze_skills"
        }

    query      = f"{state['role']} jobs in {state['location']}"
    retrieved  = compression_retriever.invoke(query)
    target_loc = state['location'].lower()
    raw_jobs   = []

    for doc in retrieved:
        doc_loc = doc.metadata.get('location', '').lower()
        # Location filter kept per user preference
        if target_loc not in doc_loc and doc_loc not in target_loc:
            continue

        content = doc.page_content.lower()
        skills  = []
        if "skills required:" in content:
            skills_line = content.split("skills required:")[-1].strip()
            skills = [
                s.strip().strip(',.')
                for s in skills_line.split(',')
                if s.strip() and len(s.strip()) > 2
            ]

        raw_jobs.append({
            "title":           doc.metadata.get('title', 'N/A'),
            "company":         doc.metadata.get('company', 'N/A'),
            "location":        doc.metadata.get('location', 'N/A'),
            "skills_required": skills,
            "seniority":       doc.metadata.get('seniority', 'N/A'),
            "link":            doc.metadata.get('source', 'N/A'),
        })

    # Deduplicate
    seen, jobs = set(), []
    for job in raw_jobs:
        key = (job['company'], tuple(sorted(s.lower() for s in job['skills_required'])))
        if key not in seen:
            seen.add(key)
            jobs.append(job)

    if not jobs:
        print("[NODE] Vector DB returned no matching jobs.")
    else:
        print(f"[NODE] Found {len(jobs)} jobs in vector DB.")

    return {
        **state,
        "jobs_found": len(jobs) > 0,
        "job_data":   jobs,
        "next_step":  "analyze_skills"
    }


def analyze_skills(state: AgentState) -> AgentState:
    """Compare user skills against job requirements and identify gaps."""
    print(f"[NODE] Analyzing skill gaps for {state['role']} in {state['location']}")

    if not state.get('job_data'):
        print("[NODE] No job data available — cannot compute skill gaps.")
        return {
            **state,
            "analysis_done": True,
            "skill_gap_data": {
                "role":                 state['role'],
                "location":             state['location'],
                "user_current_skills":  state['user_skills'],
                "all_required_skills":  [],
                "skill_gaps":           [],
                "gaps_per_job":         [],
                "average_gaps_per_job": 0,
                "warning": (
                    "No job data available. JSearch may be rate-limited or the "
                    "vector DB may be empty. Check your JSEARCH_API_KEY and "
                    "remaining quota at rapidapi.com."
                )
            },
            "next_step": "format_output"
        }

    user_normalized = {_normalize_skill(s) for s in state['user_skills']}
    all_job_skills  = set()

    for job in state['job_data']:
        all_job_skills.update(
            _normalize_skill(s)
            for s in job.get('skills_required', []) if s
        )

    user_normalized = {_normalize_skill(s) for s in state['user_skills']}

    skill_freq = Counter()
    all_job_skills = set()

    for job in state['job_data']:
        job_skills = {_normalize_skill(s) for s in job.get('skills_required', []) if s}
        all_job_skills.update(job_skills)

        for skill in job_skills:
            if skill not in user_normalized:
                skill_freq[skill] += 1

# 🔥 NEW: ranked gaps (most important first)
    top_gaps = [skill for skill, _ in skill_freq.most_common()]
    gaps_per_job    = []
    total_gap_count = 0

    for job in state['job_data']:
        job_skills = {_normalize_skill(s) for s in job.get('skills_required', []) if s}
        job_gaps   = sorted(s for s in job_skills if s not in user_normalized)
        total_gap_count += len(job_gaps)
        gaps_per_job.append({
            "title":     job.get('title', 'N/A'),
            "company":   job.get('company', 'N/A'),
            "gaps":      job_gaps,
            "gap_count": len(job_gaps)
        })

    avg_gaps = round(total_gap_count / len(state['job_data']), 1) if state['job_data'] else 0

    return {
        **state,
        "analysis_done": True,
        "skill_gap_data": {
            "role":                 state['role'],
            "location":             state['location'],
            "user_current_skills":  state['user_skills'],
            "all_required_skills":   list(all_job_skills),
            "skill_gaps":           top_gaps,
            "gaps_per_job":         gaps_per_job,
            "average_gaps_per_job": avg_gaps,
            "recommendation": (
                f"Focus on the top skill gaps to meet requirements across "
                f"{len(gaps_per_job)} {state['role']} opportunities in {state['location']}."
            )
        },
        "next_step": "format_output"
    }


def format_output(state: AgentState) -> AgentState:
    """Package final output as a JSON message."""
    print("[NODE] Formatting output...")

    sg = state.get('skill_gap_data', {})
    output = {
        "job_postings":       state['job_data'][:10],
        "skill_gap_analysis": sg,
        "summary": (
            f"Found {len(state['job_data'])} {state['role']} jobs in {state['location']}. "
            f"Identified an average of {sg.get('average_gaps_per_job', 0)} skill gaps per job."
            if state['job_data']
            else sg.get('warning', 'No job data available.')
        )
    }

    return {
        **state,
        "messages":  state["messages"] + [{"role": "assistant", "content": json.dumps(output, indent=2)}],
        "next_step": "end"
    }


def route_next_step(state: AgentState) -> Literal[
    "fetch_jobs", "search_db", "analyze_skills", "format_output", "end"
]:
    step = state.get("next_step", "end")
    print(f"[ROUTER] → {step}")
    return step


# ─────────────────────────────────────────────────────────────────────────────
# Graph — compiled at module level so supervisor can call app.invoke()
# ─────────────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    wf = StateGraph(AgentState)

    wf.add_node("fetch_jobs",     fetch_jobs)
    wf.add_node("search_db",      search_db)
    wf.add_node("analyze_skills", analyze_skills)
    wf.add_node("format_output",  format_output)

    wf.set_entry_point("fetch_jobs")

    wf.add_conditional_edges(
        "fetch_jobs", route_next_step,
        {"analyze_skills": "analyze_skills", "search_db": "search_db"}
    )
    wf.add_conditional_edges(
        "search_db", route_next_step,
        {"analyze_skills": "analyze_skills"}
    )
    wf.add_conditional_edges(
        "analyze_skills", route_next_step,
        {"format_output": "format_output"}
    )
    wf.add_conditional_edges(
        "format_output", route_next_step,
        {"end": END}
    )
    return wf


# ─────────────────────────────────────────────────────────────────────────────
# FIX #8: Keep PostgresSaver connection open for the lifetime of `app`.
# The original code used a `with` block which closed the connection right after
# compile(), making all subsequent checkpointed invocations fail.
# ─────────────────────────────────────────────────────────────────────────────

app             = None
_pg_checkpointer = None  # module-level reference keeps connection alive

try:
    from langgraph.checkpoint.postgres import PostgresSaver
    _pg_checkpointer = PostgresSaver.from_conn_string(DB_URL)
    _pg_checkpointer.setup()
    app = _build_graph().compile(checkpointer=_pg_checkpointer)
    print("[JMA] Graph compiled WITH PostgreSQL checkpointer.")
except Exception as e:
    print(f"[JMA] PostgreSQL checkpointer unavailable ({e}) — compiling without persistence.")
    app = _build_graph().compile()

print("[JMA] Job Market Analyst ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*70)
    print("JOB MARKET ANALYST — Standalone Mode")
    print("="*70 + "\n")

    role                = input("Target role (e.g. Full Stack Developer): ").strip()
    location            = input("Location (e.g. Bangalore): ").strip()
    years_of_experience = float(input("Years of experience (e.g. 0, 1.5, 3): ").strip() or "0")
    raw_skills          = input("Your skills (comma-separated): ").strip()
    user_skills         = [s.strip() for s in raw_skills.split(",") if s.strip()]

    thread_id = f"standalone_{int(time.time())}"

    initial_state = {
        "messages":            [{"role": "user", "content": f"Find {role} jobs in {location}"}],
        "role":                role,
        "location":            location,
        "user_skills":         user_skills,
        "years_of_experience": years_of_experience,
        "jobs_found":          False,
        "scraped":             False,
        "analysis_done":       False,
        "job_data":            [],
        "skill_gap_data":      {},
        "next_step":           "",
    }

    result   = app.invoke(initial_state, {"configurable": {"thread_id": thread_id}})
    last_msg = result["messages"][-1]
    print("\n" + "="*70)
    print(last_msg.content if hasattr(last_msg, 'content') else last_msg.get("content", ""))
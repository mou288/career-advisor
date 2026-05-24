# services/scraper/main.py
"""
LinkedIn Job Scraper — FastAPI service
Run with: uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""

from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import time
import spacy
from spacy.matcher import PhraseMatcher
import concurrent.futures

app = FastAPI(title="Job Scraper API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Skills list — used by spaCy PhraseMatcher to extract skills from job descriptions
# ─────────────────────────────────────────────────────────────────────────────

SKILLS = [
    # ── Languages ────────────────────────────────────────────────────────────
    "Python", "JavaScript", "TypeScript", "Java", "Go", "Golang", "Rust",
    "C", "C++", "C#", "PHP", "Ruby", "Swift", "Kotlin", "Scala", "R",
    "Julia", "Perl", "Haskell", "Elixir", "Erlang", "Clojure", "Lua",
    "Bash", "Shell", "PowerShell", "MATLAB",

    # ── Web Frontend ─────────────────────────────────────────────────────────
    "HTML", "HTML5", "CSS", "CSS3", "SASS", "SCSS", "Less",
    "React", "React.js", "Next.js", "Vue", "Vue.js", "Nuxt.js",
    "Angular", "AngularJS", "Svelte", "SvelteKit", "Remix", "Astro", "Gatsby",
    "Redux", "Zustand", "MobX", "Recoil", "Pinia",
    "Tailwind", "TailwindCSS", "Bootstrap", "Material UI", "Chakra UI",
    "Styled Components", "Emotion",
    "Webpack", "Vite", "Parcel", "Rollup", "Babel", "ESLint",
    "Jest", "Vitest", "Cypress", "Playwright", "Testing Library",
    "React Native", "Expo",

    # ── Web Backend ──────────────────────────────────────────────────────────
    "Django", "Flask", "FastAPI", "Starlette", "Tornado", "aiohttp",
    "Express", "Express.js", "NestJS", "Fastify", "Koa", "Hapi",
    "Spring", "Spring Boot", "Spring Framework",
    "Laravel", "Symfony", "CodeIgniter",
    "Ruby on Rails", "Sinatra",
    "Gin", "Fiber", "Echo", "Gorilla",
    "Actix", "Axum", "Rocket",
    "ASP.NET", ".NET", "dotnet",
    "GraphQL", "REST", "RESTful", "gRPC", "WebSocket", "Socket.IO",
    "Microservices", "Serverless",

    # ── Databases ────────────────────────────────────────────────────────────
    "PostgreSQL", "MySQL", "MariaDB", "SQLite", "Oracle", "SQL Server",
    "MongoDB", "Mongoose", "DynamoDB", "Firestore", "CouchDB",
    "Redis", "Memcached", "Cassandra", "Elasticsearch", "OpenSearch",
    "InfluxDB", "TimescaleDB", "Neo4j",
    "Prisma", "SQLAlchemy", "TypeORM", "Sequelize", "Hibernate",
    "SQL", "NoSQL",

    # ── Cloud & DevOps ───────────────────────────────────────────────────────
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
    "Nginx", "Apache", "Traefik", "Caddy",
    "Vault", "Istio", "Envoy", "Linkerd",
    "CI/CD", "DevOps", "SRE",

    # ── ML / AI ──────────────────────────────────────────────────────────────
    "PyTorch", "TensorFlow", "Keras", "JAX", "Flax",
    "scikit-learn", "XGBoost", "LightGBM", "CatBoost",
    "HuggingFace", "Transformers", "Diffusers",
    "LangChain", "LlamaIndex", "OpenAI", "Anthropic",
    "MLflow", "Weights & Biases", "Optuna", "Ray",
    "ONNX", "TorchServe", "Triton", "BentoML",
    "NLP", "Computer Vision", "Deep Learning", "Machine Learning",
    "Reinforcement Learning", "LLM", "Generative AI",
    "BERT", "GPT", "Stable Diffusion",

    # ── Data Engineering ─────────────────────────────────────────────────────
    "Apache Spark", "PySpark", "Hadoop", "Hive", "Flink",
    "Apache Kafka", "Kafka", "RabbitMQ", "Celery",
    "Apache Airflow", "Prefect", "Dagster", "Luigi",
    "dbt", "Fivetran", "Airbyte",
    "Snowflake", "BigQuery", "Redshift", "Databricks", "Delta Lake",
    "Athena", "Presto", "Trino",
    "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly",
    "Tableau", "Power BI", "Looker", "Metabase", "Superset",
    "Jupyter", "Spark SQL",

    # ── Tools & Practices ────────────────────────────────────────────────────
    "Git", "GitHub", "GitLab", "Bitbucket",
    "Jira", "Confluence", "Notion", "Linear",
    "Postman", "Swagger", "OpenAPI",
    "Agile", "Scrum", "Kanban",
    "TDD", "BDD", "Unit Testing", "Integration Testing",
    "Linux", "Unix",
    "Figma", "Storybook",
]

# ─────────────────────────────────────────────────────────────────────────────
# spaCy setup
# ─────────────────────────────────────────────────────────────────────────────

nlp     = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(skill.lower()) for skill in SKILLS]
matcher.add("SKILLS", patterns)


def extract_skills(text: str) -> list[str]:
    doc     = nlp(text.lower())
    matches = matcher(doc)
    extracted = [doc[start:end].text for _, start, end in matches]
    # Dedupe preserving original casing from SKILLS list
    seen   = set()
    result = []
    for s in extracted:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            # Return the canonical casing from our SKILLS list
            canonical = next((sk for sk in SKILLS if sk.lower() == key), s.title())
            result.append(canonical)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Core scraping functions
# ─────────────────────────────────────────────────────────────────────────────

def fetchJobDescription(job_description_url: str) -> dict | int:
    """Fetch and parse a single LinkedIn job description page."""
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                requests.get, job_description_url,
                timeout=15
            )
            desc_response = future.result()

        if desc_response.status_code == 429:
            return 429
        if desc_response.status_code != 200:
            raise Exception(f"HTTP {desc_response.status_code}")

    except Exception as e:
        raise Exception(f"Failed to retrieve job description: {e}")

    soup = BeautifulSoup(desc_response.content, "html.parser")

    # Extract job description text
    desc_div = soup.select_one(
        "div.description__text.description__text--rich > section > div"
    )
    job_text = desc_div.get_text(separator=" ", strip=True) if desc_div else ""
    skills_required = extract_skills(job_text)

    # Extract job criteria (seniority, type, function, industries)
    job_criteria = soup.find("ul", class_="description__job-criteria-list")

    def get_criteria(nth: int) -> str:
        if not job_criteria:
            return "N/A"
        el = job_criteria.select_one(f"li:nth-child({nth}) > span")
        return el.get_text(strip=True) if el else "N/A"

    return {
        "link":             job_description_url,
        "skills required":  skills_required,
        "seniority level":  get_criteria(1),
        "employment type":  get_criteria(2),
        "job function":     get_criteria(3),
        "industries":       get_criteria(4),
    }


def fetchJobs(role: str, location: str, start: int) -> list[dict] | int:
    """Fetch job listing page from LinkedIn."""
    try:
        role_enc     = role.replace(" ", "%20")
        location_enc = location.replace(" ", "%20")
        url = (
            f"https://in.linkedin.com/jobs/search"
            f"?keywords={role_enc}&location={location_enc}"
            f"&geoId=&position=1&pageNum=0&start={start}"
        )
        response = requests.get(url, timeout=15)

        if response.status_code == 429:
            return 429
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")

    except Exception as e:
        raise Exception(f"Failed to retrieve job listings: {e}")

    soup     = BeautifulSoup(response.content, "html.parser")
    all_jobs = []

    for card in soup.select("ul > li > div.base-search-card"):
        try:
            title_el    = card.find("h3", class_="base-search-card__title")
            company_el  = card.select_one("div.base-search-card__info > h4 > a")
            link_el     = card.select_one("a")
            location_el = card.select_one("div.base-search-card__info > div > span")
            date_el     = card.select_one("div > time")

            all_jobs.append({
                "title":     title_el.get_text(strip=True)    if title_el    else "N/A",
                "company":   company_el.get_text(strip=True)  if company_el  else "N/A",
                "link":      link_el.get("href")              if link_el     else None,
                "location":  location_el.get_text(strip=True) if location_el else "N/A",
                "list_date": date_el.get_text(strip=True)     if date_el     else "N/A",
            })
        except Exception:
            continue

    return all_jobs


# ─────────────────────────────────────────────────────────────────────────────
# API endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint — job_market_analyst pings this before scraping."""
    return {"status": "ok", "service": "job-scraper"}


@app.get("/api/jobs")
async def getJobs(role: str, location: str, page: int = 1):
    """
    Fetch job listings for a given role and location.
    Retries up to 20 times on LinkedIn rate-limit (429).
    """
    start = (page - 1) * 25
    for attempt in range(20):
        try:
            data = fetchJobs(role, location, start)
            if data == 429:
                time.sleep(2)
                continue
            return JSONResponse(content=data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=429, detail="Rate limited by LinkedIn after 20 retries")


@app.get("/api/jobs/description")
async def getJobDescription(url: str):
    """
    Fetch detailed description and skill requirements for a single job URL.
    Retries up to 20 times on LinkedIn rate-limit (429).
    """
    for attempt in range(20):
        try:
            data = fetchJobDescription(url)
            if data == 429:
                time.sleep(2)
                continue
            return JSONResponse(content=data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=429, detail="Rate limited by LinkedIn after 20 retries")
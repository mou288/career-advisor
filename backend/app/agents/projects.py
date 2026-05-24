"""
projects.py

Handles:
- GitHub repo fetching
- README extraction
- Project selection
- Description enhancement using LLM

No dependency on resume_strategist logic.
"""

import requests
import base64
import concurrent.futures
import re
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage
import os

GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")


# ─────────────────────────────────────────────
# GitHub Fetch
# ─────────────────────────────────────────────

def fetch_github_profile(username: str, timeout: int = 10) -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    try:
        r = requests.get(
            f"https://api.github.com/users/{username}",
            headers=headers,
            timeout=timeout
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def fetch_github_repos(username: str, per_page: int = 20, timeout: int = 10) -> List[dict]:
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
                "name": repo.get("name", ""),
                "description": repo.get("description") or "",
                "language": repo.get("language") or "",
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0),
                "html_url": repo.get("html_url", ""),
                "topics": repo.get("topics", []),
                "owner": repo.get("owner", {}).get("login", username)
            }
            for repo in r.json()
        ]
    except Exception:
        return []


# ─────────────────────────────────────────────
# README Extraction
# ─────────────────────────────────────────────

def fetch_repo_readme(owner: str, repo_name: str, timeout: int = 10) -> str:
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    readme_names = ["README.md", "README", "readme.md", "readme"]

    for name in readme_names:
        try:
            url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{name}"
            r = requests.get(url, headers=headers, timeout=timeout)

            if r.status_code == 200:
                data = r.json()
                content = data.get("content", "")
                if content:
                    decoded = base64.b64decode(content).decode("utf-8", errors="ignore")
                    return decoded[:3000]
        except Exception:
            continue

    return ""


# ─────────────────────────────────────────────
# Combined GitHub Data
# ─────────────────────────────────────────────

def fetch_github_projects_data(username: str) -> Dict[str, Any]:
    if not username:
        return {"profile": {}, "repos": []}

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            pf = ex.submit(fetch_github_profile, username)
            rp = ex.submit(fetch_github_repos, username)

            profile = pf.result()
            repos = rp.result()

        enriched_repos = []

        for repo in repos[:10]:
            readme = fetch_repo_readme(repo["owner"], repo["name"])
            repo_copy = repo.copy()
            repo_copy["readme"] = readme
            enriched_repos.append(repo_copy)

        enriched_repos.extend(repos[10:])

        return {"profile": profile, "repos": enriched_repos}

    except Exception:
        return {"profile": {}, "repos": []}


# ─────────────────────────────────────────────
# Description Enhancement (LLM)
# ─────────────────────────────────────────────

def enhance_project_description(repo, llm):
    readme = repo.get("readme", "")
    desc = repo.get("description", "")
    name = repo.get("name", "")
    lang = repo.get("language", "")

    if not readme:
        return desc or f"Developed {name} using {lang}"

    prompt = f"""
Create a strong resume project description.

Repo: {name}
Tech: {lang}
Description: {desc}
README: {readme[:1500]}

Return 2–3 sentences max.
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip()

        if len(text) > 400:
            text = text[:400]

        return text
    except Exception:
        return desc or f"Developed {name} using {lang}"


# ─────────────────────────────────────────────
# Project Selection
# ─────────────────────────────────────────────

def select_best_projects(repos, target_role, llm, max_projects=4):

    if not repos:
        return []

    try:
        summary = [
            {
                "index": i,
                "name": r["name"],
                "description": r["description"],
                "language": r["language"],
                "stars": r["stars"]
            }
            for i, r in enumerate(repos)
        ]

        prompt = f"""
Select best {max_projects} projects for role: {target_role}

{summary}

Return JSON list of indexes.
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        raw = re.sub(r"```.*?```", "", response.content, flags=re.DOTALL)
        indexes = eval(raw.strip())

    except Exception:
        repos = sorted(repos, key=lambda x: x["stars"], reverse=True)
        indexes = list(range(min(max_projects, len(repos))))

    selected = []

    for i in indexes[:max_projects]:
        if i < len(repos):
            r = repos[i]
            enhanced = enhance_project_description(r, llm)

            selected.append({
                "name": r["name"],
                "description": enhanced,
                "tech_stack": [r["language"]] if r["language"] else [],
                "stars": r["stars"],
                "html_url": r["html_url"],
                "readme": r.get("readme", "")
            })

    return selected


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────

def extract_projects(github_username: str, target_role: str, llm) -> List[Dict[str, Any]]:
    """
    Main entry point
    """
    data = fetch_github_projects_data(github_username)
    repos = data.get("repos", [])

    projects = select_best_projects(
        repos=repos,
        target_role=target_role,
        llm=llm
    )

    return projects
# curriculum_architect.py
"""
Curriculum Architect Agent
- Takes skill_gap_analysis + resume_inventory from supervisor
- Iterates through each skill gap, fetching/searching courses
- Sequential evaluation: prerequisites → interest matching → workload
- Architect LLM assembles final PersonalizedLearningPath
- Critic reviews with up to 2 revision cycles
- Exposes: app (compiled LangGraph graph)

LLM CALLS PER RUN:
  N x web_search_and_store  (one per skill with no DB hit — web search path only)
  1 x architect             (assemble final path, up to 2 retries)
  1 x critic                (review, up to 2 revision cycles)
  Total: ~3-5 LLM calls for a typical run

CHANGES FROM ORIGINAL:
  - Merged web_search_for_courses + process_and_store → web_search_and_store
  - Merged aggregate_scores + select_courses → aggregate_and_select
  - Removed prepare_revision_node (logic inlined into architect_node)
  - Fixed parallel fan-out bug: check_prerequisites → match_interests → estimate_workload (sequential)
  - Fixed missing filtered = [] in fetch_course_catalog
  - Removed unused PlannerOutput / planner chain
  - structured_critique ownership moved solely to critic_node
  - handle_submission_node no longer writes structured_critique
"""

import os
import operator
import uuid
import json
from typing import List, Optional, TypedDict, Annotated, Dict, Any
from pprint import pformat

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv

from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Shared components from settings.py
# ─────────────────────────────────────────────────────────────────────────────
from settings import llm, embedding_function

TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY")
COURSE_CHROMA_DIR = os.getenv("COURSE_CHROMA_DIR", "./chroma_db_courses")

# ─────────────────────────────────────────────────────────────────────────────
# Tavily search tool (optional)
# ─────────────────────────────────────────────────────────────────────────────
tavily_tool = None
if TavilySearchResults is not None and TAVILY_API_KEY:
    try:
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
        tavily_tool = TavilySearchResults(max_results=5)
        print("[CA] Tavily search tool ready.")
    except Exception as e:
        print(f"[CA] WARNING: Tavily unavailable: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Course vector store
# ─────────────────────────────────────────────────────────────────────────────
vectorstore = Chroma(
    persist_directory=COURSE_CHROMA_DIR,
    embedding_function=embedding_function,
    collection_name="course_catalog"
)
print(f"[CA] Course vector store ready at {COURSE_CHROMA_DIR}.")

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class Course(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    prereqs: List[int] = Field(default_factory=list)
    difficulty: int = Field(..., ge=1, le=5)
    topics: List[str] = Field(default_factory=list)
    url: str
    description: str

class NewCourseList(BaseModel):
    courses: List[Course]

class UserProfile(BaseModel):
    interests: List[str]
    skill_level: str  # Beginner | Intermediate | Advanced
    past_courses: List[int]

class CourseRecommendation(BaseModel):
    skill_name: str
    course_title: str
    course_url: str
    description: str

class PersonalizedLearningPath(BaseModel):
    user_summary: str
    skill_gaps: List[str]
    recommendations: List[CourseRecommendation]

class Critique(BaseModel):
    is_approved: bool
    revisions_needed: str


def _list_overwrite(existing: Optional[List], new: Optional[List]) -> List:
    """Reducer: [] sentinel resets the list, otherwise appends."""
    if new == []:        return []
    if existing is None: existing = []
    if new is None:      return existing
    return existing + new


class CurriculumGraphState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────────
    skill_gap_analysis: str
    resume_inventory:   str

    # ── Per-skill iteration ───────────────────────────────────────────────────
    current_skill:      Optional[str]
    course_catalog:     List[Course]

    # ── Evaluation ────────────────────────────────────────────────────────────
    evaluation_logs:    Annotated[List[Dict[str, Any]], _list_overwrite]
    all_found_courses:  Annotated[List[Dict[str, Any]], operator.add]
    evaluated_courses:  List[Dict[str, Any]]

    # ── Architect / Critic ────────────────────────────────────────────────────
    messages:            Annotated[List[BaseMessage], operator.add]
    draft_path:          Optional[PersonalizedLearningPath]
    structured_critique: Optional[Critique]
    revision_count:      Annotated[int, operator.add]


# ─────────────────────────────────────────────────────────────────────────────
# LLM chains
# ─────────────────────────────────────────────────────────────────────────────

process_search_chain = ChatPromptTemplate.from_messages([
    ("system",
     "Convert web search results into Course objects.\n"
     "- Unique IDs, realistic difficulty (1-5), prerequisites ([0] if none).\n"
     "- Must have valid URL and description. Skip non-courses.\n"
     "Output a NewCourseList object."),
    ("human", "Search results for '{skill}':\n{search_results}")
]) | llm.with_structured_output(NewCourseList)

critic_chain = ChatPromptTemplate.from_messages([
    ("system",
     "Review this learning path. Approve if:\n"
     "- Each skill gap has at least 1 recommendation.\n"
     "- Summary is clear (2-3 sentences).\n"
     "- Structure is valid.\n"
     "Reject only if >50% of gaps have no course, or structure is broken.\n"
     "Output a Critique object."),
    ("human", "Learning path:\n{draft_path_json}")
]) | llm.with_structured_output(Critique)

@tool("submit_learning_path")
def submit_learning_path(path: dict) -> str:
    """Submit the finalised learning path.
    Args must be: {"path": {"user_summary": "...", "skill_gaps": [...], "recommendations": [...]}}
    """
    if not isinstance(path, dict):
        raise ValueError("Expected path to be a dict.")
    return "Learning path submitted successfully."

model_with_tools = llm.bind_tools([submit_learning_path])

ARCHITECT_SYSTEM = """You are the Curriculum Architect assembling a PersonalizedLearningPath.

Call submit_learning_path with EXACTLY this structure:
{
  "path": {
    "user_summary": "2-3 sentences about the learner.",
    "skill_gaps": ["skill1", "skill2", ...],
    "recommendations": [
      {"skill_name": "...", "course_title": "...", "course_url": "...", "description": "..."}
    ]
  }
}

Rules: always wrap in "path", include ALL courses provided, never rename fields."""


# ─────────────────────────────────────────────────────────────────────────────
# Nodes (9 total, down from 13)
# ─────────────────────────────────────────────────────────────────────────────

def iteration_controller_node(state: CurriculumGraphState) -> dict:
    """Pop next skill or signal completion."""
    skills = [s.strip() for s in state.get("skill_gap_analysis", "").split("\n") if s.strip()]
    if not skills:
        print(f"[CA] All skills processed. {len(state.get('all_found_courses', []))} courses collected.")
        return {"current_skill": None}
    skill = skills.pop(0)
    print(f"[CA] Processing skill: {skill} ({len(skills)} remaining)")
    return {
        "current_skill":      skill,
        "skill_gap_analysis": "\n".join(skills),
        "evaluation_logs":    [],   # reset logs for this skill
        "course_catalog":     [],
    }


def fetch_course_catalog(state: CurriculumGraphState) -> dict:
    """Semantic search + fuzzy filter in ChromaDB. No LLM call."""
    skill    = state["current_skill"]
    docs     = vectorstore.similarity_search(skill, k=10)
    keywords = skill.lower().split()
    courses  = []

    for doc in docs:
        meta = doc.metadata.copy()
        if isinstance(meta.get("prereqs"), str):
            meta["prereqs"] = [int(p) for p in meta["prereqs"].split(",") if p]
        if isinstance(meta.get("topics"), str):
            meta["topics"]  = [t.strip() for t in meta["topics"].split(",") if t]
        courses.append(Course(**meta))

    # FIX: filtered was missing initialisation — caused NameError in original
    filtered = []
    for c in courses:
        text        = (c.title + " " + c.description + " " + " ".join(c.topics or [])).lower()
        match_count = sum(1 for kw in keywords if kw in text)
        if len(keywords) == 1:
            if keywords[0] in c.title.lower().split():
                filtered.append(c)
        else:
            if match_count >= max(1, len(keywords) // 2):
                filtered.append(c)

    print(f"[CA] DB: {len(courses)} found, {len(filtered)} after filter for '{skill}'")
    return {"course_catalog": filtered}


# MERGED: web_search_for_courses + process_and_store → web_search_and_store
def web_search_and_store(state: CurriculumGraphState) -> dict:
    """
    Tavily web search + LLM extraction + ChromaDB storage in one node.
    Replaces the original two-node web_search_for_courses → process_and_store chain.
    They were always sequential with no branching between them, so merging is safe.
    """
    skill = state["current_skill"]

    # ── 1. Web search ─────────────────────────────────────────────────────────
    if tavily_tool is None:
        return {"course_catalog": []}
    try:
        results = tavily_tool.invoke({"query": f"best online courses {skill} udemy coursera edx 2024"})
        print(f"[CA] Web search: {len(results)} results for '{skill}'")
    except Exception as e:
        print(f"[CA] Web search error: {e}")
        return {"course_catalog": []}

    if not results:
        return {"course_catalog": []}

    # ── 2. LLM extraction ─────────────────────────────────────────────────────
    try:
        obj         = process_search_chain.invoke({"skill": skill, "search_results": pformat(results)})
        new_courses = obj.courses
    except Exception as e:
        print(f"[CA] Course extraction error: {e}")
        return {"course_catalog": []}

    # ── 3. Store in ChromaDB ──────────────────────────────────────────────────
    if new_courses:
        docs = []
        for c in new_courses:
            meta            = c.model_dump()
            meta["prereqs"] = ",".join(map(str, meta["prereqs"]))
            meta["topics"]  = ",".join(meta["topics"])
            docs.append(Document(
                page_content=f"Course: {c.title}. Topics: {', '.join(c.topics)}. Description: {c.description}",
                metadata=meta,
            ))
        try:
            vectorstore.add_documents(docs, ids=[c.id for c in new_courses])
            print(f"[CA] Stored {len(new_courses)} new courses in DB.")
        except Exception as e:
            print(f"[CA] DB store error: {e}")

    return {"course_catalog": new_courses}


def check_prerequisites(state: CurriculumGraphState) -> dict:
    """Step 1 of sequential evaluation chain."""
    profile = state.get("user_profile") or UserProfile(
        interests=[], skill_level="Intermediate", past_courses=[]
    )
    past = set(profile.past_courses)
    logs = [
        {"type": "prereq", "course_id": c.id, "pass": set(int(p) for p in c.prereqs).issubset(past)}
        for c in state["course_catalog"]
    ]
    return {"evaluation_logs": logs}


def match_interests(state: CurriculumGraphState) -> dict:
    """Step 2 of sequential evaluation chain."""
    profile = state.get("user_profile") or UserProfile(
        interests=[], skill_level="Intermediate", past_courses=[]
    )
    interests = {i.lower() for i in profile.interests}
    logs = []
    for c in state["course_catalog"]:
        overlap = interests & {t.lower() for t in c.topics}
        score   = min(len(overlap) / len(interests) * 1.5, 1.0) if overlap else 0.5
        logs.append({"type": "interest", "course_id": c.id, "score": round(score, 2)})
    return {"evaluation_logs": logs}


def estimate_workload(state: CurriculumGraphState) -> dict:
    """Step 3 of sequential evaluation chain."""
    profile = state.get("user_profile") or UserProfile(
        interests=[], skill_level="Intermediate", past_courses=[]
    )
    max_diff = {"Beginner": 2, "Intermediate": 4, "Advanced": 5}.get(profile.skill_level, 4)
    logs = [
        {"type": "workload", "course_id": c.id, "load_ok": c.difficulty <= max_diff}
        for c in state["course_catalog"]
    ]
    return {"evaluation_logs": logs}


# MERGED: aggregate_scores + select_courses → aggregate_and_select
def aggregate_and_select(state: CurriculumGraphState) -> dict:
    """
    Combine evaluation logs into per-course scores, then select top-3.
    Replaces the original two-node aggregate_scores → select_courses chain.
    select_courses was a thin slice/reformat on top of aggregate output — no reason to split.
    """
    skill   = state["current_skill"]
    catalog = {c.id: c for c in state["course_catalog"]}

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg: Dict[str, dict] = {}
    for log in state["evaluation_logs"]:
        cid = log["course_id"]
        if cid not in agg:
            agg[cid] = {}
        if log["type"] == "prereq":    agg[cid]["prereq_pass"]    = log["pass"]
        elif log["type"] == "interest": agg[cid]["interest_score"] = log["score"]
        elif log["type"] == "workload": agg[cid]["workload_ok"]    = log["load_ok"]

    evaluated = []
    for cid, data in agg.items():
        if cid not in catalog:
            continue
        score = (
            data.get("interest_score", 0) * 10
            if data.get("prereq_pass") and data.get("workload_ok") else 0
        )
        evaluated.append({"course": catalog[cid], "score": round(score, 2), "details": data})

    # ── Select top-3 ──────────────────────────────────────────────────────────
    sorted_evals = sorted(evaluated, key=lambda x: x["score"], reverse=True)
    top          = [e for e in sorted_evals if e["score"] > 0][:2] or sorted_evals[:2]

    recs = [{
        "skill_name":   skill,
        "course_title": e["course"].title,
        "course_url":   e["course"].url,
        "description":  f"{e['course'].description} (score: {e['score']}/10)",
    } for e in top]

    print(f"[CA] Selected {len(recs)} courses for '{skill}'")
    return {"all_found_courses": recs}


def architect_node(state: CurriculumGraphState) -> dict:
    """
    LLM call — assemble final learning path and call submit tool.
    Revision feedback (previously in prepare_revision_node) is inlined here:
    when revision_count > 0, the critique message is appended before re-invoking.
    """
    revision = state.get("revision_count", 0)
    print(f"[CA] Architect running (revision {revision})...")

    courses  = state["all_found_courses"]
    by_skill = {}
    for c in courses:
        by_skill.setdefault(c.get("skill_name", "Unknown"), []).append(c.get("course_title"))

    if revision == 0:
        # Fresh run — build message list from scratch
        messages = [
            SystemMessage(content=ARCHITECT_SYSTEM),
            HumanMessage(content=(
                f"Resume info:\n{state.get('resume_inventory', '{}')}\n\n"
                f"Courses ({len(courses)} total):\n{pformat(courses)}\n\n"
                f"Include ALL {len(courses)} courses. {len(by_skill)} skill gaps."
            )),
        ]
    else:
        # Revision run — append critique feedback to existing message history
        # (replaces the removed prepare_revision_node)
        critique = state.get("structured_critique")
        feedback = critique.revisions_needed if critique else "Resubmit with all required fields."
        messages = state["messages"] + [
            HumanMessage(content=f"Feedback:\n{feedback}\n\nResubmit using submit_learning_path.")
        ]

    for attempt in range(2):
        response = model_with_tools.invoke(messages)
        if getattr(response, "tool_calls", None):
            print(f"[CA] Architect tool called on attempt {attempt + 1}")
            return {"messages": [response]}
        if attempt == 0:
            messages = messages + [response, HumanMessage(content="Call the submit_learning_path tool.")]

    return {"messages": [response]}


def handle_submission_node(state: CurriculumGraphState) -> dict:
    """
    Parse tool call from architect and validate against Pydantic schema.
    No longer writes structured_critique — critic_node owns that field exclusively.
    On parse failure, sets a submission_error flag and increments revision_count
    so the graph routes back to architect_node directly.
    """
    last = state["messages"][-1]
    tcs  = getattr(last, "tool_calls", None)

    if not tcs:
        print("[CA] Architect did not call tool — requesting revision.")
        return {
            "draft_path":     None,
            "messages":       [ToolMessage(content="Tool not called.", tool_call_id="error")],
            "revision_count": 1,
        }

    tc   = tcs[0]
    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
    args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)

    if name != "submit_learning_path":
        return {}

    try:
        if isinstance(args, str):
            args = json.loads(args)
        path_data = args.get("path") or args
        if isinstance(path_data, str):
            path_data = json.loads(path_data)
        draft = PersonalizedLearningPath(**path_data)
        tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "tool")
        print(f"[CA] Draft saved: {len(draft.recommendations)} recommendations.")
        return {
            "draft_path": draft,
            "messages":   [ToolMessage(content="Submitted.", tool_call_id=tc_id)],
        }
    except Exception as e:
        print(f"[CA] Submission parse error: {e}")
        # FIX: no longer sets structured_critique here — that's critic_node's job.
        # revision_count increment routes back to architect via handle_submission conditional.
        return {
            "draft_path":     None,
            "messages":       [ToolMessage(content=f"Error: {e}", tool_call_id="error")],
            "revision_count": 1,
        }


def critic_node(state: CurriculumGraphState) -> dict:
    """
    LLM call — review draft path.
    Sole owner of structured_critique in the graph.
    """
    result = critic_chain.invoke({"draft_path_json": state["draft_path"].model_dump_json(indent=2)})
    status = "APPROVED" if result.is_approved else "NEEDS REVISION"
    print(f"[CA] Critic: {status}")
    return {"structured_critique": result, "revision_count": 1}


# ─────────────────────────────────────────────────────────────────────────────
# Build and compile graph
# ─────────────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    wf = StateGraph(CurriculumGraphState)

    # 9 nodes (down from 13)
    wf.add_node("iteration_controller",  iteration_controller_node)
    wf.add_node("fetch_course_catalog",  fetch_course_catalog)
    wf.add_node("web_search_and_store",  web_search_and_store)       # merged
    wf.add_node("check_prerequisites",   check_prerequisites)
    wf.add_node("match_interests",       match_interests)
    wf.add_node("estimate_workload",     estimate_workload)
    wf.add_node("aggregate_and_select",  aggregate_and_select)       # merged
    wf.add_node("architect",             architect_node)              # inlined prepare_revision
    wf.add_node("handle_submission",     handle_submission_node)
    wf.add_node("critic",                critic_node)

    wf.set_entry_point("iteration_controller")

    # ── Iteration loop ────────────────────────────────────────────────────────
    wf.add_conditional_edges(
        "iteration_controller",
        lambda s: "architect" if s.get("current_skill") is None else "fetch_course_catalog",
        {"architect": "architect", "fetch_course_catalog": "fetch_course_catalog"},
    )

    # ── DB hit or web search ──────────────────────────────────────────────────
    wf.add_conditional_edges(
        "fetch_course_catalog",
        lambda s: "web_search_and_store" if len(s.get("course_catalog", [])) < 3 else "check_prerequisites",
        {"web_search_and_store": "web_search_and_store", "check_prerequisites": "check_prerequisites"},
    )

    wf.add_edge("web_search_and_store", "check_prerequisites")

    # ── Sequential evaluation chain (FIX: was broken parallel fan-out) ────────
    wf.add_edge("check_prerequisites",  "match_interests")
    wf.add_edge("match_interests",      "estimate_workload")
    wf.add_edge("estimate_workload",    "aggregate_and_select")

    wf.add_edge("aggregate_and_select", "iteration_controller")

    # ── Architect → submission ────────────────────────────────────────────────
    wf.add_edge("architect", "handle_submission")

    wf.add_conditional_edges(
        "handle_submission",
        lambda s: "critic" if s.get("draft_path") else "architect",
        {"critic": "critic", "architect": "architect"},
    )

    # ── Critic → end or revision ──────────────────────────────────────────────
    wf.add_conditional_edges(
        "critic",
        lambda s: (
            END if s["structured_critique"].is_approved or s.get("revision_count", 0) >= 2
            else "architect"
        ),
        {"architect": "architect", END: END},
    )

    return wf


app = _build_graph().compile()
print("[CA] Curriculum Architect graph compiled and ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    skill_gaps = input("Skill gap analysis (Enter for example): ").strip() or """
angular, aws, docker, kubernetes, react, css, html, vue.js
"""
    resume = input("Resume inventory (Enter for example): ").strip() or """
Target Role: Full Stack Developer | Skills: Python, Node.js, MongoDB, PostgreSQL
"""
    result = app.invoke(
        {"skill_gap_analysis": skill_gaps, "resume_inventory": resume},
        {"recursion_limit": 200},
    )
    path = result.get("draft_path")
    if path:
        print(f"\n{path.user_summary}\n")
        for rec in path.recommendations:
            print(f"  [{rec.skill_name}] {rec.course_title} — {rec.course_url}")
    else:
        print("No path generated.", result.get("structured_critique"))
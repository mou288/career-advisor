# interview_coach.py (interview_agent_v3_interactive.py)
import os
from typing import TypedDict, List, Literal, Annotated, Optional, Dict, Any
import operator
import random
from datetime import datetime
import re
import uuid 
import psycopg

from langgraph.types import Command, interrupt
from langchain_core.messages import ( 
    BaseMessage, 
    SystemMessage, 
    HumanMessage, 
    AIMessage
)

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_cohere import CohereRerank
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever

# --- Project Imports ---
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from settings import llm, embedding_function, DB_URL, COHERE_API_KEY

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INTERVIEW_CHROMA_DIR = os.path.join(SCRIPT_DIR, "ingestion", "chroma_db_interview_questions")
MAX_QUESTIONS_PER_SESSION = 15
MESSAGE_HISTORY_LIMIT = 10
USE_IN_MEMORY_CHECKPOINTER = False

# --- Project question probability (configurable) ---
PROJECT_QUESTION_PROBABILITY = 0.25  # 25% chance to ask project question

# --- Skill to Topic Mapping ---
SKILL_TO_TOPIC_MAP = {
    # Frontend Frameworks
    "react": "frameworks (frontend)",
    "reactjs": "frameworks (frontend)",
    "vue": "frameworks (frontend)",
    "vuejs": "frameworks (frontend)",
    "angular": "frameworks (frontend)",
    "angularjs": "frameworks (frontend)",
    "nextjs": "frameworks (frontend)",
    "next.js": "frameworks (frontend)",
    "nuxt": "frameworks (frontend)",
    "nuxtjs": "frameworks (frontend)",
    "svelte": "frameworks (frontend)",
    "ember": "frameworks (frontend)",
    "backbone": "frameworks (frontend)",
    
    # Backend Frameworks
    "django": "frameworks (backend)",
    "flask": "frameworks (backend)",
    "fastapi": "frameworks (backend)",
    "express": "frameworks (backend)",
    "expressjs": "frameworks (backend)",
    "nodejs": "frameworks (backend)",
    "node.js": "frameworks (backend)",
    "node": "frameworks (backend)",
    "rails": "frameworks (backend)",
    "ruby on rails": "frameworks (backend)",
    "laravel": "frameworks (backend)",
    "spring": "frameworks (backend)",
    "spring boot": "frameworks (backend)",
    "nestjs": "frameworks (backend)",
    "koa": "frameworks (backend)",
    "gin": "frameworks (backend)",
    "echo": "frameworks (backend)",
    "fiber": "frameworks (backend)",
    
    # Programming Languages
    "python": "programming languages",
    "javascript": "programming languages",
    "js": "programming languages",
    "typescript": "programming languages",
    "ts": "programming languages",
    "go": "programming languages",
    "golang": "programming languages",
    "rust": "programming languages",
    "java": "programming languages",
    "c++": "programming languages",
    "cpp": "programming languages",
    "c#": "programming languages",
    "csharp": "programming languages",
    "ruby": "programming languages",
    "php": "programming languages",
    "kotlin": "programming languages",
    "swift": "programming languages",
    "scala": "programming languages",
    
    # Databases
    "postgresql": "databases & data engineering",
    "postgres": "databases & data engineering",
    "mongodb": "databases & data engineering",
    "mongo": "databases & data engineering",
    "redis": "databases & data engineering",
    "mysql": "databases & data engineering",
    "sqlite": "databases & data engineering",
    "cassandra": "databases & data engineering",
    "dynamodb": "databases & data engineering",
    "elasticsearch": "databases & data engineering",
    "mariadb": "databases & data engineering",
    "oracle": "databases & data engineering",
    "mssql": "databases & data engineering",
    "sql server": "databases & data engineering",
    "neo4j": "databases & data engineering",
    "couchdb": "databases & data engineering",
    
    # Infrastructure & DevOps
    "docker": "deployment & server management",
    "kubernetes": "deployment & server management",
    "k8s": "deployment & server management",
    "aws": "deployment & server management",
    "azure": "deployment & server management",
    "gcp": "deployment & server management",
    "google cloud": "deployment & server management",
    "terraform": "deployment & server management",
    "ansible": "deployment & server management",
    "jenkins": "deployment & server management",
    "gitlab": "deployment & server management",
    "github actions": "deployment & server management",
    "circleci": "deployment & server management",
    "travis": "deployment & server management",
    "nginx": "deployment & server management",
    "apache": "deployment & server management",
    "serverless": "deployment & server management",
    "lambda": "deployment & server management",
    
    # APIs & Web Services
    "graphql": "restful apis & web services",
    "rest": "restful apis & web services",
    "restful": "restful apis & web services",
    "api": "restful apis & web services",
    "grpc": "restful apis & web services",
    "websocket": "restful apis & web services",
    "soap": "restful apis & web services",
    
    # Data Structures & Algorithms (general)
    "algorithms": "data structures & algorithms",
    "data structures": "data structures & algorithms",
    "leetcode": "data structures & algorithms",
    "competitive programming": "data structures & algorithms",
    
    # System Design
    "microservices": "system design",
    "distributed systems": "system design",
    "architecture": "system design",
    "scalability": "system design",
    "load balancing": "system design",
    "caching": "system design",
    
    # Security
    "oauth": "concurrency, networking & security",
    "jwt": "concurrency, networking & security",
    "authentication": "concurrency, networking & security",
    "authorization": "concurrency, networking & security",
    "encryption": "concurrency, networking & security",
    "ssl": "concurrency, networking & security",
    "tls": "concurrency, networking & security",
    
    # Testing
    "jest": "programming languages",
    "pytest": "programming languages",
    "junit": "programming languages",
    "mocha": "programming languages",
    "testing": "programming languages",
    
    # Message Queues
    "rabbitmq": "system design",
    "kafka": "system design",
    "celery": "frameworks (backend)",
    "redis queue": "frameworks (backend)",
    
    # Frontend Build Tools
    "webpack": "frameworks (frontend)",
    "vite": "frameworks (frontend)",
    "rollup": "frameworks (frontend)",
    "babel": "frameworks (frontend)",
}


# --- 1. State Definition ---
class InterviewAgentState(MessagesState):
    """Extended MessagesState for interview-specific fields."""
    user_id: str
    interview_focus: Literal["all", "proficient", "gaps"]
    company_focus: str
    topic_list: List[str]
    
    evaluation_report: List[dict]
    covered_questions: List[str]
    
    current_question_id: str
    current_question_topic: str  
    current_question_difficulty: str  
    start_time: datetime
    question_count: int
    topic_performance: Dict[str, List[dict]]  
    current_difficulty_per_topic: Dict[str, List[str]]
    follow_up_count: int
    user_context: Dict[str, Any]


print("\n" + "="*70)
print("INITIALIZING LANGGRAPH v1.0 DEPENDENCIES")
print("="*70)

# Vector DB
try:
    interview_db = Chroma(
        persist_directory=INTERVIEW_CHROMA_DIR,
        embedding_function=embedding_function
    )
    count = interview_db._collection.count()
    print(f"Interview DB: {count} questions loaded")
except Exception as e:
    print(f"ChromaDB Error: {e}")
    raise RuntimeError(f"Failed to load interview DB: {e}")

# Reranker
cohere_reranker = None
if COHERE_API_KEY:
    try:
        cohere_reranker = CohereRerank(
            cohere_api_key=COHERE_API_KEY, 
            model="rerank-english-v3.0"
        )
        print("CohereRerank initialized")
    except Exception as e:
        print(f"Reranker Warning: {e}")
else:
    print("No COHERE_API_KEY set. Reranker disabled.")

print("="*70)

# --- 3. LLM Prompts ---
EVALUATION_PROMPT = PromptTemplate(
    input_variables=["question", "answer"],
    template="""You are a senior technical interviewer. Evaluate this answer:

Question: {question}
Candidate Answer: {answer}

Provide:
GRADE: [bad/moderate/good/excellent]
SKILL: [primary technical skill tested]
FEEDBACK: [2-3 sentences]"""
)

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["report"],
    template="""Summarize this interview performance:

{report}

Provide a concise summary."""
)

ACTION_CHOICE_PROMPT = PromptTemplate(
    input_variables=["question", "answer", "grade"],
    template="""You are an interview agent's routing logic.

The user was asked: {question}
The user answered: {answer}
The evaluation was: {grade}

Based on this evaluation, should you:
1. Ask a probing follow-up question (because the answer was incomplete, vague, or 'moderate').
2. Move on to a completely new question (because the answer was 'excellent', 'good', or 'bad' and unsalvageable).

Ask follow up question strictly when its necessary, avoid unnecessary follow up questions.

Important : Respond with ONLY the word "FOLLOW_UP" or "NEW_QUESTION"."""
)

FOLLOW_UP_PROMPT = PromptTemplate(
    input_variables=["question", "answer", "grade"],
    template="""You are a technical interviewer.

Original Question: {question}
Candidate's Answer: {answer}
Your internal evaluation: {grade}

Your goal is to probe deeper. Ask a SINGLE, concise follow-up question based on their answer.
Do NOT say "Good answer" or "Okay". Just ask the follow-up question.

Example: If the answer was about 'sharding', a good follow-up is 'How would you handle a hot shard?'

Your follow-up question:"""
)

# --- Helper Functions ---

def normalize_skills_to_topics(github_skills: List[str]) -> List[str]:
    """
    Map specific technologies from GitHub to broader interview topics.
    Returns deduplicated list of interview-relevant topics.
    """
    normalized = set()
    unmapped_skills = []
    
    for skill in github_skills:
        skill_lower = skill.lower().strip()
        normalized.add(skill_lower)
        
        if skill_lower in SKILL_TO_TOPIC_MAP:
            normalized.add(SKILL_TO_TOPIC_MAP[skill_lower])
        else:
            # Keep unmapped skills for semantic search fallback
            unmapped_skills.append(skill_lower)
    
    result = list(normalized)
    
    print(f"\n[SKILL MAPPING]")
    print(f"  Input skills: {github_skills}")
    print(f"  Mapped to topics: {result}")
    if unmapped_skills:
        print(f"  Unmapped (will use semantic search): {unmapped_skills}")
    
    return result


def get_next_difficulty_state(current_diffs: List[str], upgrade: bool = False, downgrade: bool = False) -> List[str]:
    """Return next difficulty state based on transition."""
    
    current_set = set(current_diffs)

    if upgrade:
        if current_set != {"hard"}:
            return ["hard", "medium"]
        else:
            return current_diffs
        
    elif downgrade:
        if current_set != {"easy"}:
            return ["easy", "medium"]
        else:
            return current_diffs
        
    return current_diffs


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Project Question Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_project_question(
    state: InterviewAgentState,
    user_projects: List[Dict[str, Any]]
) -> dict:
    """
    Generate a personalized question about user's GitHub project.
    Uses project README and description to create specific, technical questions.
    """
    if not user_projects:
        return {}
    
    # Select a random project that hasn't been covered recently
    covered_project_names = [
        q_id[len("project_"):-9]  # strips "project_" prefix and "_xxxxxxxx" uuid suffix
        for q_id in state.get("covered_questions", [])
        if q_id.startswith("project_")
]
    
    available_projects = [
        p for p in user_projects 
        if p.get("name", "").lower().replace(" ", "-") not in covered_project_names

    ]
    
    # If all projects covered, allow any
    if not available_projects:
        available_projects = user_projects
    
    project = random.choice(available_projects)
    
    #project_name = project.get("name", "your project")
    project_name = project.get("name", "").strip() or "your project"
    description = project.get("description", "")
    readme = project.get("readme", "")
    tech_stack = project.get("tech_stack", [])
    stars = project.get("stars", 0)
    
    # Build rich context for LLM
    project_context = f"""
Project: {project_name}
Description: {description}
Tech Stack: {', '.join(tech_stack) if tech_stack else 'Not specified'}
Stars: {stars}
README (excerpt): {readme[:1500] if readme else 'No README available'}
"""
    
    # Generate question using LLM
    prompt = f"""You are a technical interviewer conducting a coding interview.

Generate ONE specific, technical question about this candidate's GitHub project:

{project_context}

Question Requirements:
1. Must mention the project name: "{project_name}"
2. Ask about technical decisions, architecture, or challenges
3. Be open-ended (not yes/no)
4. Require detailed explanation
5. Cannot be answered by just reading the README
6. Focus on "why" and "how" rather than "what"

Good Examples:
- "In {project_name}, how did you handle state management across components?"
- "What led you to choose {tech_stack[0] if tech_stack else 'that architecture'} for {project_name}?"
- "Can you walk me through a challenging bug you encountered in {project_name}?"
- "How would you scale {project_name} to handle 10x more users?"

Bad Examples (too generic):
- "Tell me about your project"
- "What does this do?"

Output ONLY the question, no preamble or explanation."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        question = response.content.strip()
        print(f"[INTERVIEW] Generated project question for: {project_name}")
    except Exception as e:
        print(f"[INTERVIEW] Project question generation failed: {e}")
        # Fallback to template-based question
        if tech_stack:
            question = f"In your {project_name} project, what led you to choose {tech_stack[0]} and how did it impact your development process?"
        else:
            question = f"Can you walk me through the technical architecture of your {project_name} project and any interesting challenges you faced?"
    
    # Get current state
    messages = state.get("messages", [])
    if len(messages) > MESSAGE_HISTORY_LIMIT:
        messages = messages[-MESSAGE_HISTORY_LIMIT:]
    
    # Return in same format as regular questions
    return {
        "messages": messages + [AIMessage(content=question, name="Interviewer")],
        "current_question_id": f"project_{project_name.lower().replace(' ', '-')}_{str(uuid.uuid4())[:8]}",
        "current_question_topic": "projects",
        "current_question_difficulty": "medium",
        "question_count": state.get("question_count", 0) + 1,
        "follow_up_count": 0,
        "current_difficulty_per_topic": state.get("current_difficulty_per_topic", {}),
}


# ─────────────────────────────────────────────────────────────────────────────
# Graph Nodes
# ─────────────────────────────────────────────────────────────────────────────

def start_interview_node(state: InterviewAgentState) -> dict:
    """Initialize interview session."""
    print("\n" + "="*70)
    print("INTERVIEW SESSION STARTING")
    print("="*70)
    
    if not state.get("topic_list"):
        raise ValueError("topic_list is required")
    
    current_difficulty_per_topic = {
        topic.lower(): ["easy", "medium", "hard"]
        for topic in state["topic_list"]
    }
    
    print(f"Initial difficulty settings: {current_difficulty_per_topic}")
    
    return {
        "messages": [
            SystemMessage(
                content=(
                    f"Interview Configuration\n"
                    f"Focus: {state['interview_focus']}\n"
                    f"Company: {state['company_focus']}\n"
                    f"Topics: {', '.join(state['topic_list'])}"
                )
            )
        ],
        "start_time": datetime.now(),
        "question_count": 0,
        "current_question_id": "",
        "current_question_topic": "",
        "current_question_difficulty": "",
        "covered_questions": [],
        "evaluation_report": [],
        "topic_performance": {},
        "follow_up_count": 0,
        "current_difficulty_per_topic": current_difficulty_per_topic,
        "user_context": state.get("user_context", {})
    }


def generate_question_node(state: InterviewAgentState) -> dict:
    """
    RAG node to fetch the next question, with project question injection.
    Now supports GitHub project-based questions and enhanced skill mapping.
    """
    print(f"\n--- generate_question (Q#{state.get('question_count', 0) + 1}) ---")

    # --- PROJECT QUESTION INJECTION ---
    user_projects = state.get("user_context", {}).get("projects", [])
    
    # Count how many project questions we've asked
    # ADD this instead:
    project_question_count = sum(
    1 for q_id in state.get("covered_questions", [])
    if q_id.startswith("project_")
)
    question_count = state.get("question_count", 0)

# Criteria enforcement
    all_projects_covered_once = project_question_count >= len(user_projects)
    at_max_cap = project_question_count >= len(user_projects) * 2
    is_last_few_questions = question_count >= MAX_QUESTIONS_PER_SESSION - len(user_projects) -2

# Force project question if projects exist but haven't been asked yet
# and we're running out of time
    must_ask_project = (
    user_projects and
    not all_projects_covered_once and
    is_last_few_questions
)

# Pure random otherwise, gated by cap
    random_says_ask = (
    user_projects and
    not at_max_cap and          # never Q1
    random.random() < PROJECT_QUESTION_PROBABILITY
)

    should_ask_project = must_ask_project or random_says_ask
    if should_ask_project:
        print("[INTERVIEW] Generating project-based question...")
        project_result = generate_project_question(state, user_projects)
        if project_result:  # If successful
            return project_result
        # If failed, continue to regular question generation below
    # --- END PROJECT QUESTION INJECTION ---
    
    topics = state["topic_list"]
    topics = [t for t in topics if t != "projects"]
    focus = state["interview_focus"]
    company = state["company_focus"]
    covered_ids = state.get("covered_questions", [])
    
    # --- Adaptive Difficulty Logic ---
    topic_performance = state.get("topic_performance", {})
    current_difficulty_per_topic = state.get("current_difficulty_per_topic", {})
    difficulty_updates = {}
    messages_to_print = []
    
    for topic_original_case in topics:
        topic = topic_original_case.lower()
        
        recent_evals = topic_performance.get(topic, [])
        curr_diffs = current_difficulty_per_topic.get(topic, ["easy", "medium", "hard"])
        
        # Downgrade Logic
        hard_evals = [e for e in recent_evals if e["difficulty"] == "hard"]
        if len(hard_evals) >= 2:
            last_two_hard = hard_evals[-2:]
            
            if all(e["evaluation"] == "bad" for e in last_two_hard):
                new_diffs = get_next_difficulty_state(curr_diffs, downgrade=True)
                if new_diffs != curr_diffs:
                    messages_to_print.append(f"\n{'='*70}\n>>> Difficulty decreased for {topic_original_case}: {curr_diffs} → {new_diffs}\n{'='*70}")
                    difficulty_updates[topic] = new_diffs

        # Upgrade Logic
        easy_evals = [e for e in recent_evals if e["difficulty"] == "easy"]
        
        if len(easy_evals) >= 2 and topic not in difficulty_updates:
            last_two_easy = easy_evals[-2:]
            
            if all(e["evaluation"] == "excellent" for e in last_two_easy):
                new_diffs = get_next_difficulty_state(curr_diffs, upgrade=True)
                if new_diffs != curr_diffs:
                    messages_to_print.append(f"\n{'='*70}\n>>> Difficulty increased for {topic_original_case}: {curr_diffs} → {new_diffs}\n{'='*70}\n{'='*70}")
                    difficulty_updates[topic] = new_diffs

    # Apply difficulty updates
    updated_difficulty_map = {**current_difficulty_per_topic, **difficulty_updates}
    
    # Print difficulty change notifications
    for msg in messages_to_print:
        print(msg)
    
    # --- Build Company Filter ---
    all_companies_list = ["amazon", "microsoft", "nvidia", "netflix", "meta", "google", "generic"]
    
    if company == "all":
        companies_to_search = all_companies_list
    elif company == "generic":
        companies_to_search = ["generic"]
    else:
        companies_to_search = [company]
    
    print(f"Searching for companies: {companies_to_search}")

    # Aggregate allowed difficulties across all topics
    allowed_difficulties = set()
    for topic_original_case in topics:
        topic = topic_original_case.lower()
        diffs = updated_difficulty_map.get(topic, ["easy", "medium", "hard"])
        allowed_difficulties.update(diffs)
    
    allowed_difficulties = list(allowed_difficulties)
    
    # --- Enhanced Query Construction with User Context ---
    user_context = state.get("user_context", {})
    github_skills = user_context.get("skills", [])
    role = user_context.get("role", "Software Engineer")
    
    query_text = f"""{role} interview at {company}.
Topics: {', '.join(topics)}
Related technologies: {', '.join(github_skills[:10]) if github_skills else 'general'}
Looking for questions about: system design, coding problems, framework knowledge, real-world scenarios"""
    

    
    # --- TWO-STAGE RETRIEVAL ---
    
    # STAGE 1: Strict Filter (High Precision)
    strict_filter_conditions = [
        {"topic": {"$in": [t.lower() for t in topics]}},
        {"company": {"$in": companies_to_search}},
        {"difficulty": {"$in": allowed_difficulties}},
    ]
    
    if covered_ids:
        strict_filter_conditions.append({"question_id": {"$nin": covered_ids}})
    
    strict_metadata_filter = {"$and": strict_filter_conditions}
    
    #print(f"[STAGE 1] Strict filter: {strict_metadata_filter}")
    
    strict_results = []
    try:
        base_retriever = interview_db.as_retriever(
            search_kwargs={"k": 10, "filter": strict_metadata_filter}
        )
        
        if cohere_reranker:
            #print("  → Using Cohere reranker...")
            compressor = ContextualCompressionRetriever(
                base_compressor=cohere_reranker, 
                base_retriever=base_retriever
            )
            strict_results = compressor.invoke(query_text)
        else:
            print("  → Using vector search...")
            strict_results = base_retriever.invoke(query_text)
        
        print(f"  → Found {len(strict_results)} using stage 1 questions")
    except Exception as e:
        print(f"  → Stage 1 failed: {e}")
    
    # STAGE 2: Semantic Fallback (High Recall)
    if len(strict_results) < 3:
        #print(f"\n[STAGE 2] Only {len(strict_results)} results from strict filter. Using semantic search...")
        
        # Remove topic filter, rely on vector similarity
        semantic_filter_conditions = [
            {"company": {"$in": companies_to_search}},
            {"difficulty": {"$in": allowed_difficulties}},
        ]
        
        if covered_ids:
            semantic_filter_conditions.append({"question_id": {"$nin": covered_ids}})
        
        semantic_metadata_filter = {"$and": semantic_filter_conditions}
        
        #print(f"  Semantic filter: {semantic_metadata_filter}")
        
        try:
            base_retriever = interview_db.as_retriever(
                search_kwargs={"k": 30, "filter": semantic_metadata_filter}  # Cast wider net
            )
            
            if cohere_reranker:
                print("  → Using Cohere reranker (semantic)...")
                compressor = ContextualCompressionRetriever(
                    base_compressor=cohere_reranker, 
                    base_retriever=base_retriever
                )
                semantic_results = compressor.invoke(query_text)
            else:
                print("  → Using vector search (semantic)...")
                semantic_results = base_retriever.invoke(query_text)
            
            print(f"  → Found {len(semantic_results)} questions using stage 2")
            
            # Merge results (strict first, then semantic)
            seen_ids = {r.metadata.get("question_id") for r in strict_results}
            combined_results = strict_results + [
                r for r in semantic_results 
                if r.metadata.get("question_id") not in seen_ids
            ]
            
            results = combined_results[:15]
            print(f"  → Combined total: {len(results)} questions")
            
        except Exception as e:
            print(f"  → Stage 2 failed: {e}")
            results = strict_results
    else:
        results = strict_results
    
   

    # LLM GENERATION FALLBACK
    if not results:
        #print("\n[LLM FALLBACK] No questions found in database. Generating question via LLM...")

        skill = random.choice(topics) if topics else "general software engineering"
        difficulty = random.choices(
            ["easy", "medium", "hard"],
            weights=[0.2, 0.6, 0.2]
        )[0]

        prompt = f"""You are a technical interviewer at {company}.

Generate ONE realistic technical interview question.

Role: {role}
Skill/Topic: {skill}
Difficulty: {difficulty}
Context: This candidate has experience with: {', '.join(github_skills[:5]) if github_skills else 'various technologies'}

Rules:
- Must be specific and realistic (not generic)
- Should test practical knowledge, not just theory
- No need to provide the answer
- Make it relevant to {company} if possible

Output ONLY the question, no preamble."""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            question = response.content.strip()
            
        except Exception as e:
            print(f"  → LLM generation failed: {e}")
            question = f"Explain how you would approach {skill} in a production environment at {company}."

        # Trim message history
        messages = state.get("messages", [])
        if len(messages) > MESSAGE_HISTORY_LIMIT:
            messages = messages[-MESSAGE_HISTORY_LIMIT:]

        return {
            "messages": messages + [AIMessage(content=question, name="Interviewer")],
            "current_question_id": f"llm_gen_{uuid.uuid4()}",
            "current_question_topic": skill.lower(),
            "current_question_difficulty": difficulty,
            "question_count": state.get("question_count", 0) + 1,
            "current_difficulty_per_topic": updated_difficulty_map,
            "follow_up_count": 0,
        }
    
    # --- SELECT QUESTION FROM RESULTS ---
    # Select randomly from top 3 for variety
    N = min(3, len(results))
    selectable_results = results[:N]
    next_doc = random.choice(selectable_results)
    
    next_question = next_doc.page_content.strip()
    next_id = next_doc.metadata.get("question_id", f"retrieved_{uuid.uuid4()}")
    next_topic = next_doc.metadata.get("topic", "general").lower()
    next_difficulty = next_doc.metadata.get("difficulty", "medium")
    
    print(f"\n[SELECTED QUESTION]")
    print(f"  ID: {next_id}")
    print(f"  Topic: {next_topic} | Difficulty: {next_difficulty}")
    print(f"  Question: {next_question[:100]}...")
    
    # Trim message history
    messages = state.get("messages", [])
    if len(messages) > MESSAGE_HISTORY_LIMIT:
        messages = messages[-MESSAGE_HISTORY_LIMIT:]
    
    return {
        "messages": messages + [AIMessage(content=next_question, name="Interviewer")],
        "current_question_id": next_id,
        "current_question_topic": next_topic,
        "current_question_difficulty": next_difficulty,
        "question_count": state.get("question_count", 0) + 1,
        "current_difficulty_per_topic": updated_difficulty_map,
        "follow_up_count": 0,
    }


def ask_follow_up_node(state: InterviewAgentState) -> dict:
    """Generate LLM-based follow-up question."""
    
    
    last_eval = state.get("evaluation_report", [])[-1]
    question = last_eval["question"]
    answer = last_eval["answer"]
    grade = last_eval["evaluation"]
    
    try:
        prompt = FOLLOW_UP_PROMPT.format(question=question, answer=answer, grade=grade)
        response = llm.invoke([HumanMessage(content=prompt)])
        follow_up_question = response.content.strip()
    except Exception as e:
        print(f"Follow-up generation failed: {e}")
        follow_up_question = "Can you elaborate on that a bit more?"
    
    print(f"Asking follow-up: {follow_up_question}")
    
    messages = state["messages"]
        
    return {
        "messages": messages + [AIMessage(content=follow_up_question, name="Interviewer")],
        "follow_up_count": state.get("follow_up_count", 0) + 1,
    }


def wait_for_answer_node(state: InterviewAgentState) -> dict:
    
    resumed_value = interrupt(value={"type": "wait_for_input"})

    # HARD STOP (this prevents auto continuation)
    if resumed_value is None:
        return {}

    

    return {
        "messages": resumed_value.get("messages", [])
    }




def wait_for_followup_node(state: InterviewAgentState) -> dict:
    """Separate wait node for follow-ups — prevents parallel execution branching."""
    
    resumed_value = interrupt(value={"type": "wait_for_input"})
    if resumed_value is None:
        return {}
    
    return {"messages": resumed_value.get("messages", [])}


def evaluate_answer_node(state: InterviewAgentState) -> dict:
    """Evaluate the user's answer."""
    
    
    messages = state["messages"]
    
    # Find question-answer pair
    question_msg = answer_msg = None
    for i, msg in enumerate(reversed(messages)):
        if isinstance(msg, HumanMessage) and msg.name != "Interviewer":
            answer_msg = msg
            for j in range(len(messages) - i - 2, -1, -1):
                if isinstance(messages[j], AIMessage) and messages[j].name == "Interviewer":
                    question_msg = messages[j]
                    break
            break
    
    if not question_msg or not answer_msg:
        raise ValueError("Q/A pair not found — state corrupted")
    
    # Get metadata from state
    current_q_id = state["current_question_id"]
    topic = state.get("current_question_topic", "general").lower()
    difficulty = state.get("current_question_difficulty", "medium")
    
    # LLM Evaluation
    try:
        prompt = EVALUATION_PROMPT.format(
            question=question_msg.content,
            answer=answer_msg.content
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        
        grade = re.search(r"GRADE:\s*(.+)", response.content, re.IGNORECASE)
        skill = re.search(r"SKILL:\s*(.+)", response.content, re.IGNORECASE)
        feedback = re.search(r"FEEDBACK:\s*(.+)", response.content, re.IGNORECASE | re.DOTALL)
        
        grade = grade.group(1).strip().lower() if grade else "moderate"
        skill = skill.group(1).strip().lower() if skill else "general"
        feedback = feedback.group(1).strip() if feedback else "No feedback."
        
        print(f"Grade: {grade.upper()} | Skill: {skill} | Topic: {topic} | Difficulty: {difficulty}")

        # Print feedback immediately
        print("\n" + "="*40 + "\nFEEDBACK" + "="*40)
        print(f"Grade: {grade.upper()}\nFeedback: {feedback}")
        print("="*40)

    except Exception as e:
        print(f"Evaluation failed: {e}")
        grade, skill, feedback = "error", "unknown", str(e)

    # State update logic
    current_report = state.get("evaluation_report", []).copy()
    topic_performance = state.get("topic_performance", {}).copy()
    if topic not in topic_performance:
        topic_performance[topic] = []

    # Check if this is a follow-up
    is_follow_up = bool(
        current_report and 
        current_report[-1]["question_id"] == current_q_id
    )

    if is_follow_up:
        print("This is a follow-up. UPDATING previous grade.")
        
        last_eval_report = current_report[-1]
        last_eval_report["evaluation"] = grade
        last_eval_report["feedback"] = feedback
        last_eval_report["answer"] = f"{last_eval_report.get('answer', '')}\n[FOLLOW-UP]: {answer_msg.content}"
        last_eval_report["timestamp"] = datetime.now().isoformat()
        
        if topic_performance.get(topic):
            last_perf_entry = topic_performance[topic][-1]
            if last_perf_entry["question_id"] == current_q_id:
                last_perf_entry["evaluation"] = grade
                last_perf_entry["timestamp"] = datetime.now().isoformat()
        
        return {
            "messages": [SystemMessage(content=f"Grade (Updated): {grade.upper()}\nFeedback: {feedback}")],
            "evaluation_report": current_report,
            "topic_performance": topic_performance
        }
        
    else:
        print("This is a new question. APPENDING to report.")
        
        new_eval_entry = {
            "question_id": current_q_id,
            "question": question_msg.content,
            "answer": answer_msg.content,
            "evaluation": grade,
            "feedback": feedback,
            "skill": skill,
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "difficulty": difficulty
        }
        
        topic_performance[topic].append({
            "question_id": current_q_id,
            "evaluation": grade,
            "difficulty": difficulty,
            "timestamp": datetime.now().isoformat()
        })
        topic_performance[topic] = topic_performance[topic][-5:]

        return {
            "messages": [SystemMessage(content=f"Grade: {grade.upper()}\nFeedback: {feedback}")],
            "evaluation_report": current_report + [new_eval_entry],
            "topic_performance": topic_performance
        }

    
def update_coverage_node(state: InterviewAgentState) -> dict:
    print("--- Node: update_coverage ---")
    
    current_id = state.get("current_question_id")
    covered = state.get("covered_questions", []).copy()

    if current_id and current_id not in ["NONE", "ERROR", "DONE"] and current_id not in covered:
        print(f"Covered: {current_id}")
        covered.append(current_id)
        return {"covered_questions": covered}

    # Return a no-op that still gives LangGraph something to checkpoint
    # without modifying covered_questions
    print(f"Already covered or invalid: {current_id}")
    return {"covered_questions": state.get("covered_questions", [])}


def decide_action_router(state: InterviewAgentState) -> Literal["ask_follow_up", "get_new_question"]:
    print("--- Router: decide_action ---")
    
    MAX_FOLLOW_UPS = 2  # ← hard cap per question
    
    try:
        # Hard cap check first
        if state.get("follow_up_count", 0) >= MAX_FOLLOW_UPS:
            print(f"Follow-up limit ({MAX_FOLLOW_UPS}) reached. Moving to new question.")
            return "get_new_question"
        
        last_eval = state.get("evaluation_report", [])[-1]
        grade = last_eval["evaluation"].lower()
        
        if grade in ["excellent", "good"]:
            print("Answer was good. Moving to new question.")
            return "get_new_question"
        
        question = last_eval["question"]
        answer = last_eval["answer"]
        
        prompt = ACTION_CHOICE_PROMPT.format(question=question, answer=answer, grade=grade)
        response = llm.invoke([HumanMessage(content=prompt)])
        choice = response.content.upper().strip()
        
        if "FOLLOW_UP" in choice:
            print("LLM decided to ask a follow-up.")
            return "ask_follow_up"
        else:
            print("LLM decided to move on.")
            return "get_new_question"
            
    except Exception as e:
        print(f"Action router failed: {e}. Defaulting to new question.")
        return "get_new_question"

def should_continue_router(state: InterviewAgentState) -> Literal["generate_question", "synthesize_and_update"]:
    """Route to next node."""
    print("--- Router: should_continue ---")
    
    # Check user stop intent
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and msg.name != "Interviewer":
            content = msg.content.lower().strip()
            
            stop_triggers = ["stop", "done", "end", "finish", "quit"]
            if content in stop_triggers:
                print("User stop detected")
                return "synthesize_and_update"
            
            break
    
    # Check termination
    if state.get("current_question_id") in ["NONE", "ERROR"]:
        print("No more questions available")
        return "synthesize_and_update"
    
    if state.get("question_count", 0) >= MAX_QUESTIONS_PER_SESSION:
        print(f"Max questions ({MAX_QUESTIONS_PER_SESSION}) reached")
        return "synthesize_and_update"
    
    print("Continuing...")
    return "generate_question"


def synthesize_and_update_node(state: InterviewAgentState) -> dict:
    """Generate final summary."""
    print("\n" + "="*70)
    print("INTERVIEW SESSION COMPLETE")
    print("="*70)
    
    report = state.get("evaluation_report", [])
    if not report:
        summary = "No questions were answered in this session."
    else:
        try:
            report_str = "\n".join([
                f"- {e['skill']}: {e['evaluation']} | {e['feedback']}"
                for e in report
            ])
            prompt = SUMMARY_PROMPT.format(report=report_str)
            summary = llm.invoke(prompt).content
        except Exception as e:
            print(f"Summary error: {e}")
            summary = f"Completed {len(report)} questions."
    
    print("\nFinal Summary:\n")
    print(summary)
    
    return {
        "messages": [SystemMessage(content=f"Interview Complete!\n\n{summary}")],
        "current_question_id": "DONE"
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build Graph
# ─────────────────────────────────────────────────────────────────────────────

print("\nBuilding LangGraph Graph...")

workflow = StateGraph(InterviewAgentState)

# Add nodes
workflow.add_node("start_interview", start_interview_node)
workflow.add_node("generate_question", generate_question_node)
workflow.add_node("wait_for_answer", wait_for_answer_node) 
workflow.add_node("evaluate_answer", evaluate_answer_node)
workflow.add_node("ask_follow_up", ask_follow_up_node)
workflow.add_node("update_coverage", update_coverage_node)
workflow.add_node("synthesize_and_update", synthesize_and_update_node)
workflow.add_node("wait_for_followup", wait_for_followup_node)

# Define flow
workflow.add_edge(START, "start_interview")
workflow.add_edge("start_interview", "generate_question")

workflow.add_edge("generate_question", "wait_for_answer")
workflow.add_edge("wait_for_answer", "evaluate_answer")

workflow.add_conditional_edges(
    "evaluate_answer",
    decide_action_router,
    {
        "ask_follow_up": "ask_follow_up",
        "get_new_question": "update_coverage"
    }
)

workflow.add_edge("ask_follow_up", "wait_for_followup")
workflow.add_edge("wait_for_followup", "evaluate_answer")

workflow.add_conditional_edges(
    "update_coverage",
    should_continue_router,
    {
        "generate_question": "generate_question",
        "synthesize_and_update": "synthesize_and_update"
    }
)

workflow.add_edge("synthesize_and_update", END)


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def print_agent_output(messages: List[BaseMessage]):
    """Print only the latest AI Interviewer question."""
    
    question = None
    
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.name == "Interviewer":
            question = msg.content
            break
            
    if question:
        print("\n" + "="*40 + "\nINTERVIEWER\n" + "="*40)
        print(question)


def prompt_for_choice(prompt: str, choices: List[str]) -> str:
    """Get validated user input from a list."""
    print(f"\n{prompt}")
    for i, choice in enumerate(choices, 1):
        print(f"  {i}. {choice}")
    
    while True:
        try:
            user_input = input(f"Enter number (1-{len(choices)}): ")
            idx = int(user_input) - 1
            if 0 <= idx < len(choices):
                selected = choices[idx]
                print(f"Selected: {selected}")
                return selected
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(choices)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def derive_topics(interview_focus, user_skills, skill_gaps, has_projects=False):
    """
    Derive interview topics using skill normalization, optionally including projects.
    """
    topics = []
    
    if interview_focus == "proficient":
        # Normalize user skills to interview topics
        if user_skills:
            normalized = normalize_skills_to_topics(user_skills)
            topics = normalized[:8]
    
    elif interview_focus == "gaps":
        # Normalize skill gaps to interview topics
        if skill_gaps:
            normalized = normalize_skills_to_topics(skill_gaps)
            topics = normalized[:8]
    
    elif interview_focus == "all":
        # Combine and normalize both
        combined_skills = list(set((user_skills or []) + (skill_gaps or [])))
        if combined_skills:
            normalized = normalize_skills_to_topics(combined_skills)
            topics = normalized[:10]
    
    # Add "projects" as a meta-topic if user has projects
    if has_projects and "projects" not in topics:
        topics.append("projects")
    
    # Fallback if no topics derived
    if not topics:
        print("\n[WARNING] No topics could be derived from skills. Using fallback topics.")
        topics = ["programming languages", "data structures & algorithms", "system design"]
    
    return topics


# ─────────────────────────────────────────────────────────────────────────────
# Main Interview Function
# ─────────────────────────────────────────────────────────────────────────────

def run_interactive_interview(
    user_skills=None,
    skill_gaps=None,
    target_role=None,
    projects=None,
):
    session_id = f"interview_{str(uuid.uuid4())}"
    config = {"configurable": {"thread_id": session_id}}
    print(f"Starting new interview session: {session_id}")

    if target_role:
        print(f"\nTarget Role: {target_role}")
    if user_skills:
        print(f"Your Skills: {len(user_skills)} identified")
    if skill_gaps:
        print(f"Skill Gaps to Address: {len(skill_gaps)}")
    if projects:
        print(f"\nGitHub Projects ({len(projects)}):")
        for p in projects[:3]:
            print(f"  {p.get('name')} ({p.get('stars', 0)} stars)")

    print("\n===== PERSONALIZED INTERVIEW SETUP =====")

    if skill_gaps:
        focus_choices = ["proficient", "gaps", "all"]
    else:
        focus_choices = ["proficient", "all"]

    interview_focus = prompt_for_choice("\nChoose interview focus:", focus_choices)

    has_projects = bool(projects)
    topic_list = derive_topics(
        interview_focus,
        user_skills or [],
        skill_gaps or [],
        has_projects=has_projects
    )

    print(f"\nTopics selected ({len(topic_list)}):")
    for t in topic_list:
        print(f" - {t}")

    company_choices = ["amazon", "microsoft", "nvidia", "netflix", "meta", "google", "generic", "all"]
    company_focus = prompt_for_choice("\nChoose your company focus:", company_choices)

    initial_state = {
        "user_id": "interactive_user",
        "interview_focus": interview_focus,
        "company_focus": company_focus,
        "topic_list": topic_list,
        "messages": [],
        "user_context": {
            "skills": user_skills or [],
            "gaps": skill_gaps or [],
            "role": target_role,
            "projects": projects or []
        }
    }

    print("\nStarting interview... (Type 'stop' at any time to end)")
    if has_projects:
        print(f"  → {len(projects)} project(s) loaded - expect personalized project questions!")

    # ── KEY FIX: track the last printed question OUTSIDE the loop ──
    last_printed_question = None

    def get_and_print_latest_question():
        """Pull the latest interviewer question from state and print it exactly once."""
        nonlocal last_printed_question
        state = interview_agent.get_state(config)
        messages = state.values.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.name == "Interviewer":
                if msg.content != last_printed_question:
                    print("\n" + "=" * 40 + "\nQUESTION\n" + "=" * 40)
                    print(msg.content)
                    last_printed_question = msg.content
                return  # stop after finding the latest one

    def exhaust_stream(input_value):
        """Run stream to completion, return the last chunk."""
        last_chunk = None
        for chunk in interview_agent.stream(input_value, config=config, stream_mode="updates"):
            last_chunk = chunk   # keep iterating — never print inside here
        return last_chunk

    # ── Initial run ──
    result = exhaust_stream(initial_state)
    get_and_print_latest_question()

    # ── Interaction loop ──
    while result and "__interrupt__" in result:
        try:
            user_input = input("\nYour Answer: ").strip()

            resume_payload = {"messages": [HumanMessage(content=user_input)]}
            result = exhaust_stream(Command(resume=resume_payload))

            # Only print after the stream is fully exhausted
            get_and_print_latest_question()

        except KeyboardInterrupt:
            print("\n\nInterview interrupted by user. Ending session.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            import traceback
            traceback.print_exc()
            break

    # ── Final summary ──
    if result and "__interrupt__" not in result:
        print("\n" + "=" * 50 + "\nINTERVIEW COMPLETE\n" + "=" * 50)
        state = interview_agent.get_state(config)
        messages = state.values.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, SystemMessage) and "Interview Complete" in msg.content:
                print(msg.content)
                break# ─────────────────────────────────────────────────────────────────────────────
# Initialize Interview Agent
# ─────────────────────────────────────────────────────────────────────────────

try:
    print(f"Connecting to DB_URL: {DB_URL}")
    pg_conn = psycopg.connect(DB_URL, autocommit=True)
    interview_checkpointer = PostgresSaver(pg_conn)
    interview_checkpointer.setup()
    interview_agent = workflow.compile(checkpointer=interview_checkpointer)
    print("✅ Interview agent compiled WITH PostgresSaver + psycopg connection")
except Exception as e:
    import traceback
    print("Failed to initialize Postgres checkpointer!")
    traceback.print_exc()
    interview_checkpointer = None
    interview_agent = workflow.compile()
    print("⚠ Running WITHOUT checkpointer (resume will NOT work!)")

print("CHECKPOINTER USED:", interview_agent.checkpointer)

if __name__ == "__main__":
    # Example: Run with sample data
    sample_skills = ["react", "nodejs", "postgresql", "docker", "aws"]
    sample_gaps = ["kubernetes", "microservices", "graphql"]
    sample_projects = [
        {
            "name": "ecommerce-platform",
            "description": "Full-stack e-commerce platform with React and Node.js",
            "tech_stack": ["react", "nodejs", "postgresql", "redis"],
            "stars": 42,
            "readme": "A scalable e-commerce platform built with modern web technologies..."
        }
    ]
    
    run_interactive_interview(
        user_skills=sample_skills,
        skill_gaps=sample_gaps,
        target_role="Full Stack Engineer",
        projects=sample_projects
    )
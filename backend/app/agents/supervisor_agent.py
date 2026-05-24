# backend/app/agents/supervisor_agent.py
import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"
import sys
import io
import time
import json
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from skills import get_final_skills_data
from projects import extract_projects
from settings import llm



# Ensure utf-8 on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Set up paths
ROOT = Path(__file__).resolve().parents[3]
AGENTS_DIR = Path(__file__).parent

for path in [str(ROOT), str(AGENTS_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from datetime import datetime
from enum import Enum

# Colors / UI helpers
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

    @staticmethod
    def header(text: str):
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{text.center(80)}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}\n")

    @staticmethod
    def section(text: str):
        print(f"\n{Colors.BOLD}{Colors.BLUE}> {text}{Colors.END}")
        print(f"{Colors.BLUE}{'-'*80}{Colors.END}")

    @staticmethod
    def success(text: str):
        print(f"{Colors.GREEN}SUCCESS: {text}{Colors.END}")

    @staticmethod
    def error(text: str):
        print(f"{Colors.RED}ERROR: {text}{Colors.END}")

    @staticmethod
    def info(text: str):
        print(f"{Colors.YELLOW}INFO: {text}{Colors.END}")

# State Management
class SessionState(Enum):
    INITIALIZED = "initialized"
    INPUT_COLLECTED = "input_collected"
    SKILLS_EXTRACTED = "skills_extracted"
    JOB_ANALYSIS = "job_analysis"
    RESUME_GENERATED = "resume_generated"
    CURRICULUM_READY = "curriculum_ready"
    INTERVIEW_DONE = "interview_done"
    REPORT_GENERATED = "report_generated"
    CLEANED_UP = "cleaned_up"
    ERROR = "error"

class GlobalState:
    def __init__(self):
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = time.time()
        self.state = SessionState.INITIALIZED
        self.state_history: List[Tuple[str, float]] = []
        self.target_role: Optional[str] = None
        self.location: Optional[str] = None
        self.resume_path: Optional[str] = None
        self.github_username: Optional[str] = None
        self.email: Optional[str] = None
        self.phone: Optional[str] = None
        self.full_name: Optional[str] = None
        self.years_of_experience: int = 0
        self.education: List[Dict[str, Any]] = []
        self.work_experience: List[Dict[str, Any]] = []
        self.projects: List[Dict[str, Any]] = []
        self.user_skills: List[str] = []
        self.job_data: Dict[str, Any] = {}
        self.skill_gaps: List[str] = []
        self.learning_path: Optional[Any] = None
        self.resume_json: Optional[Dict[str, Any]] = None
        self.resume_inventory: Optional[str] = None
        self.resume_pdf_path: Optional[str] = None
        self.interview_history: List[Dict[str, Any]] = []
        self.final_report: Optional[str] = None
        self.errors: List[str] = []

    def update_state(self, new_state: SessionState, message: str = ""):
        self.state = new_state
        elapsed = time.time() - self.start_time
        self.state_history.append((new_state.value + (": " + message if message else ""), elapsed))
        Colors.info(f"[{new_state.value}] {message}")

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.update_state(SessionState.ERROR, msg)
        Colors.error(msg)

# Enhanced Import Handler
import importlib.util
from pathlib import Path

def safe_import(module_name: str):
    """Bulletproof import: loads module from same folder as supervisor_agent.py"""
    try:
        base_dir = Path(__file__).parent   # folder where supervisor_agent.py is
        
        module_path = base_dir / f"{module_name}.py"
        
        if not module_path.exists():
            print(f"INFO: Module '{module_name}' not found at {module_path}")
            return None
        
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print(f"SUCCESS: Loaded module '{module_name}' from {module_path}")
        return module
    
    except Exception as e:
        print(f"ERROR: Failed to import {module_name}: {e}")
        return None
# Agent Wrappers

def extract_skills_wrapper(state: GlobalState) -> bool:
    """Extract skills from resume or GitHub"""
    Colors.section("Extracting Skills")

    if state.user_skills:
        Colors.info("Skills already extracted. Skipping.")
        return True
    try:
        skills_mod = safe_import('skills')
        if not skills_mod or not hasattr(skills_mod, 'get_final_skills_data'):
            Colors.info("Skills module not found; skipping extraction.")
            return False

        if state.resume_path:
            Colors.info(f"Extracting skills from resume: {state.resume_path}")
            skills_set = skills_mod.get_final_skills_data(None, state.resume_path)

        elif state.github_username:
            Colors.info(f"Extracting skills from GitHub: {state.github_username}")
            skills_set = skills_mod.get_final_skills_data(state.github_username, None)
            
  

        else:
            Colors.info("No resume or GitHub provided.")
            return False

        state.user_skills = sorted(list(skills_set)) if skills_set else []
        Colors.success(f"Extracted {len(state.user_skills)} skills")
        
        if state.user_skills:
            print("\nYour Skills:")
            for i in range(0, len(state.user_skills), 3):
                chunk = state.user_skills[i:i+3]
                print("  " + " | ".join(f"{s:<20}" for s in chunk))
        
        state.update_state(SessionState.SKILLS_EXTRACTED, f"Extracted {len(state.user_skills)}")
        return True

    except Exception as e:
        traceback.print_exc()
        state.add_error(f"Skill extraction failed: {e}")
        return False
    

def extract_projects_wrapper(state: GlobalState) -> bool:
    """Extract projects from GitHub"""

    if state.projects:
        Colors.info("Projects already extracted. Skipping.")
        return True
    Colors.section("Extracting Projects")

    try:
        if not state.github_username:
            Colors.info("No GitHub provided; skipping project extraction.")
            return False

        state.projects = extract_projects(
            github_username=state.github_username,
            target_role=state.target_role,
            llm=llm
        )

        Colors.success(f"Extracted {len(state.projects)} projects")
        state.update_state(SessionState.INPUT_COLLECTED, "Projects extracted")

        if state.projects:
            print("\nYour Projects:")
            for i, p in enumerate(state.projects[:5], 1):
                print(f"  {i}. {p.get('name', '')}")

        return True

    except Exception as e:
        traceback.print_exc()
        state.add_error(f"Project extraction failed: {e}")
        return False   


def job_market_analysis_wrapper(state: GlobalState) -> bool:
    """Run job market analysis via the compiled LangGraph app."""
    Colors.section("Running Job Market Analysis")
    try:
        jm = safe_import('job_market_analyst')
        if not jm:
            Colors.info("job_market_analyst module not found; skipping.")
            return False

        if not hasattr(jm, 'app') or jm.app is None:
            Colors.error("job_market_analyst.app not available — graph failed to compile.")
            return False

        role     = state.target_role or "Software Engineer"
        location = state.location    or "Remote"

        Colors.info(f"Searching for {role} jobs in {location}...")

        thread_id = f"supervisor_{state.session_id}"

        initial_state = {
            "messages":       [{"role": "user", "content": f"Find {role} jobs in {location}"}],
            "role":           role,
            "location":       location,
            "user_skills":    state.user_skills or [],
            "scraper_up":     False,
            "jobs_found":     False,
            "scraped":        False,
            "analysis_done":  False,
            "job_data":       [],
            "skill_gap_data": {},
            "next_step":      "",
        }

        result = jm.app.invoke(
            initial_state,
            {"configurable": {"thread_id": thread_id}}
        )

        sg               = result.get('skill_gap_data', {})
        state.skill_gaps = sg.get('skill_gaps', []) if sg else []
        state.job_data   = result.get('job_data', [])

        # Warn user clearly if no data came back
        if not state.job_data:
            warning = sg.get('warning', '')
            if warning:
                Colors.error(f"No job data: {warning}")
            else:
                Colors.info("No jobs found for this role and location.")

        Colors.success("Job market analysis complete")
        state.update_state(
            SessionState.JOB_ANALYSIS,
            f"Jobs: {len(state.job_data)}, Gaps: {len(state.skill_gaps)}"
        )

        if state.skill_gaps:
            print("\nTop Skill Gaps:")
            for i, g in enumerate(state.skill_gaps[:10], 1):
                print(f"  {i}. {g}")
        else:
            Colors.info("\nGreat news! No skill gaps detected based on current job market analysis.")

        return True
    except Exception as e:
        traceback.print_exc()
        state.add_error(f"Job analysis failed: {e}")
        return False

def collect_resume_details(state: GlobalState):
    """Collect contact info, education, and work experience from the user.
    Only called once — when the user asks to generate a resume.
    Skips fields already populated on state."""
    Colors.section("Resume Details")
    print("I need a few details to build your resume.\n")

    # Contact
    if not state.full_name:
        state.full_name = input("Full name: ").strip() or "Candidate"
    if not state.email:
        state.email = input("Email: ").strip()
    if not state.phone:
        state.phone = input("Phone: ").strip()
    if not state.years_of_experience:
        try:
            state.years_of_experience = int(input("Years of experience (e.g. 3): ").strip() or "0")
        except ValueError:
            state.years_of_experience = 0

    # Education
    if not state.education:
        print("\n--- Education (leave Institution blank to stop) ---")
        while True:
            inst = input("  Institution: ").strip()
            if not inst:
                break
            deg  = input("  Degree: ").strip()
            field = input("  Field of study: ").strip()
            status = input("  Status (Completed / In Progress): ").strip()
            state.education.append({
                "institution": inst,
                "degree": deg,
                "field": field,
                "status": status
            })

    # Work experience
    if not state.work_experience:
        print("\n--- Work Experience (leave Company blank to stop) ---")
        while True:
            comp = input("  Company: ").strip()
            if not comp:
                break
            pos = input("  Position: ").strip()
            dur = input("  Duration (e.g. Jan 2022 – Dec 2023): ").strip()
            bullets = []
            print("  Achievements / responsibilities (type 'done' to stop):")
            while len(bullets) < 4:
                b = input("    - ").strip()
                if not b or b.lower() == "done":
                    break
                bullets.append(b)
            state.work_experience.append({
                "company": comp,
                "position": pos,
                "duration": dur,
                "description_bullets": bullets
            })

    Colors.success("Resume details collected.")


def generate_resume_wrapper(state: GlobalState, force: bool = False) -> bool:
    """Generate resume using ResumeStrategist with lazy state validation"""

    Colors.section("Generating Resume")

    try:
        rs_mod = safe_import("resume_agent")

        if not rs_mod:
            Colors.error("resume_strategist module could not be imported.")
            return False

        if not hasattr(rs_mod, "ResumeStrategist"):
            Colors.error("ResumeStrategist class not found.")
            return False

        strategist = rs_mod.ResumeStrategist()

        # ✅ REQUIRED FIELD CHECKS

        if not state.target_role:
            state.target_role = input("Target role: ").strip()

        if not state.user_skills:
            Colors.info("No skills found. Extracting skills first...")
            if not extract_skills_wrapper(state):
                Colors.error("Cannot generate resume without skills.")
                return False

        # ✅ OPTIONAL: PROJECTS (enhancement only)
        if not state.projects:
            if state.github_username:
                Colors.info("Extracting projects from GitHub...")
                extract_projects_wrapper(state)
            else:
                Colors.info("No projects found. Resume will be generated without project section.")

        # ✅ COLLECT USER DETAILS (lazy)
        collect_resume_details(state)

        Colors.info(f"Building resume for role: {state.target_role}")
        Colors.info(f"GitHub: {state.github_username or 'not provided'}")
        Colors.info(f"Resume file: {state.resume_path or 'not provided'}")

        contact = {
            "name":     state.full_name or "",
            "email":    state.email or "",
            "phone":    state.phone or "",
            "location": state.location or "",
            "github":   state.github_username or ""
        }

        # ✅ GENERATE RESUME
        result = strategist.generate_resume(
            target_role=state.target_role,
            github_username=state.github_username,
            resume_path=state.resume_path,
            user_skills=state.user_skills or [],
            contact=contact,
            education=state.education,
            work_experience=state.work_experience,
            years_of_experience=state.years_of_experience,
        )

        # ✅ STORE RESULTS
        state.resume_json      = result.get("resume_json", {})
        state.resume_inventory = result.get("resume_inventory", "")
        state.resume_pdf_path  = result.get("latex_path", "")

        # ✅ SAFE PROJECT MERGE (IMPORTANT)
        if result.get("projects"):
            existing = {p.get("name") for p in state.projects}
            for p in result["projects"]:
                if p.get("name") not in existing:
                    state.projects.append(p)

        Colors.success("Resume generated successfully!")
        state.update_state(SessionState.RESUME_GENERATED)

        print("\n==================== RESUME INVENTORY ====================\n")
        print(state.resume_inventory)
        print("\n===========================================================\n")

        return True

    except Exception as e:
        traceback.print_exc()
        state.add_error(f"Resume generation failed: {e}")
        return False
    

def interview_wrapper(state: GlobalState) -> bool:
    """Launch interactive interview WITH USER CONTEXT"""
    Colors.section("Starting Interactive Interview")

    try:
        # ✅ REQUIRED FIELDS

        if not state.target_role:
            state.target_role = input("Target role: ").strip()

        if not state.user_skills:
            Colors.info("Extracting skills first...")
            if not extract_skills_wrapper(state):
                Colors.error("Cannot start interview without skills.")
                return False

        if not state.skill_gaps:
            Colors.info("No skill gaps found. Running job analysis...")
            job_market_analysis_wrapper(state)

        # ✅ OPTIONAL (projects)
        if not state.projects:
            if state.github_username:
                Colors.info("Extracting projects from GitHub...")
                extract_projects_wrapper(state)
            else:
                Colors.info("No projects found. Interview will skip project-based questions.")

        # ✅ IMPORT INTERVIEW AGENT
        ia = safe_import('interview_coach')
        if not ia:
            Colors.error("interview_coach.py not found")
            return False

        if not hasattr(ia, 'run_interactive_interview'):
            Colors.error("run_interactive_interview function not found")
            return False

        # ✅ UI
        Colors.header("INTERVIEW COACH - Interactive Session")
        Colors.info("="*70)
        Colors.info("Interview will focus on your skill gaps and target role.")
        print("\n" + "="*70 + "\n")

        Colors.info(f"Target Role: {state.target_role}")
        Colors.info(f"Your Current Skills: {len(state.user_skills)} identified")

        if not state.skill_gaps:
            Colors.info("No specific skill gaps identified.")
            Colors.info("Interview will focus on general competencies.")

        print()

        # ✅ RUN INTERVIEW
        try:
            ia.run_interactive_interview(
                user_skills=state.user_skills or [],
                skill_gaps=state.skill_gaps or [],
                target_role=state.target_role or "Software Engineer",
                projects=state.projects or [],
            )

            state.update_state(SessionState.INTERVIEW_DONE, "Interview completed")
            Colors.success("\nInterview session completed!")
            return True

        except KeyboardInterrupt:
            Colors.info("\nInterview interrupted by user")
            state.update_state(SessionState.INTERVIEW_DONE, "Interview interrupted")
            return False

    except Exception as e:
        Colors.error(f"Interview failed: {e}")
        traceback.print_exc()
        state.add_error(f"Interview failed: {e}")
        return False
    
def assemble_and_save_report(state: GlobalState) -> str:
    """Generate final report"""
    Colors.section("Generating Final Report")
    try:
        report_lines = [
            "="*80,
            "CAREER ADVISORY REPORT".center(80),
            "="*80,
            f"Session: {state.session_id}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {int(time.time() - state.start_time)}s",
            "",
            "USER PROFILE",
            "-" * 80,
            f"Target Role: {state.target_role or 'Not specified'}",
            f"Location: {state.location or 'Not specified'}",
            f"GitHub: {state.github_username or 'Not provided'}",
            f"Skills Extracted: {len(state.user_skills)}",
            ""
        ]
        
        if state.skill_gaps:
            report_lines.extend([
                "SKILL GAPS IDENTIFIED",
                "-" * 80,
                f"Total Gaps: {len(state.skill_gaps)}",
                ""
            ])
            for i, g in enumerate(state.skill_gaps[:20], 1):
                report_lines.append(f"  {i}. {g}")
            report_lines.append("")
        else:
            report_lines.extend([
                "SKILL GAPS",
                "-" * 80,
                "No skill gaps identified! You have excellent coverage for the target role.",
                ""
            ])
        
        if state.learning_path:
            learning_path_data = state.learning_path
            
            # Handle different learning path formats
            if hasattr(learning_path_data, "recommendations"):
                recommendations = learning_path_data.recommendations
            elif isinstance(learning_path_data, dict):
                recommendations = learning_path_data.get("recommendations", [])
                if not recommendations and "path" in learning_path_data:
                    path_data = learning_path_data["path"]
                    if isinstance(path_data, dict):
                        recommendations = path_data.get("recommendations", [])
            else:
                recommendations = []
            
            if recommendations:
                report_lines.extend([
                    "RECOMMENDED COURSES",
                    "-" * 80,
                    f"Total Courses: {len(recommendations)}",
                    ""
                ])
                for i, rec in enumerate(recommendations[:15], 1):
                    if isinstance(rec, dict):
                        title = rec.get("course_title", "Unknown Course")
                        skill = rec.get("skill_name", "")
                        url = rec.get("course_url", "")
                    else:
                        title = getattr(rec, "course_title", str(rec))
                        skill = getattr(rec, "skill_name", "")
                        url = getattr(rec, "course_url", "")
                    
                    report_lines.append(f"  {i}. {title}")
                    if skill:
                        report_lines.append(f"     Skill: {skill}")
                    if url:
                        report_lines.append(f"     URL: {url}")
                    report_lines.append("")
        
        if state.resume_pdf_path:
            report_lines.extend([
                "RESUME",
                "-" * 80,
                f"Generated: {state.resume_pdf_path}",
                ""
            ])
        
        if state.interview_history:
            report_lines.extend([
                "INTERVIEW SUMMARY",
                "-" * 80
            ])
            for i, s in enumerate(state.interview_history, 1):
                report_lines.append(f"  Session {i}: {s.get('summary', s) if isinstance(s, dict) else s}")
            report_lines.append("")
        
        report_lines.extend([
            "NEXT STEPS & TIPS",
            "-" * 80,
        ])
        
        if state.skill_gaps:
            report_lines.extend([
                "1. Focus on closing the top 5 skill gaps",
                "2. Complete recommended courses in priority order",
                "3. Practice 2-3 interview questions per topic daily",
                "4. Build projects showcasing new skills",
                "5. Update your resume with new skills and projects",
            ])
        else:
            report_lines.extend([
                "1. Continue strengthening your existing skills",
                "2. Stay updated with latest industry trends",
                "3. Work on advanced projects in your domain",
                "4. Practice interview questions for your target role",
                "5. Network and apply to positions matching your profile",
            ])
        
        report_lines.extend([
            "",
            "="*80
        ])
        
        report_text = "\n".join(report_lines)
        fname = f"report_{state.session_id}.txt"
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        Colors.success(f"Report saved: {fname}")
        state.final_report = fname
        state.update_state(SessionState.REPORT_GENERATED, f"Report: {fname}")
        return fname
    except Exception as e:
        traceback.print_exc()
        state.add_error(f"Report generation failed: {e}")
        return ""

def curriculum_wrapper(state: GlobalState, use_existing_skills: bool = False) -> bool:
    """Build learning path using Curriculum Architect with lazy input handling"""

    Colors.section("Building Learning Path (Curriculum Architect)")

    try:
        # ✅ ENSURE REQUIRED FIELDS

        if not state.target_role:
            state.target_role = input("Target role: ").strip()

        if not state.years_of_experience:
            try:
                state.years_of_experience = int(input("Years of experience: ").strip() or "0")
            except:
                state.years_of_experience = 0

        # ✅ ENSURE SKILLS / GAPS

        if not (state.user_skills or state.skill_gaps):
            Colors.info("No skills found. Extracting skills first...")
            if not extract_skills_wrapper(state):
                Colors.error("Cannot build curriculum without skills.")
                return False

        # If still no gaps → run job analysis
        if not state.skill_gaps and not use_existing_skills:
            Colors.info("No skill gaps found. Running job analysis...")
            job_market_analysis_wrapper(state)

        # If still empty → fallback to skills
        if not state.skill_gaps and not use_existing_skills:
            Colors.info("No skill gaps detected. Switching to existing skills.")
            use_existing_skills = True

        # ✅ IMPORT MODULE
        cur = safe_import("curriculum_architect")
        if not cur or not hasattr(cur, "app"):
            Colors.error("Curriculum architect not found.")
            return False

        # Filter soft skills
        SOFT_SKILLS = {
            "communication", "teamwork", "collaboration", "problem solving",
            "scrum", "agile", "user experience", "user interface",
            "leadership", "presentation", "time management"
        }

        # ✅ SELECT SKILLS SOURCE
        if use_existing_skills:
            technical_skills = [
                s for s in state.user_skills if s.lower() not in SOFT_SKILLS
            ]
            Colors.info("Building advanced learning path for your existing skills.")
        else:
            technical_skills = [
                s for s in state.skill_gaps if s.lower() not in SOFT_SKILLS
            ]
        
        # Fallback
        if not technical_skills:
            technical_skills = state.user_skills or state.skill_gaps

        if not technical_skills:
            Colors.error("No skills available to build curriculum.")
            return False
        
        selected_skills = technical_skills[:5]

        skill_gap_text = "\n".join(selected_skills)

        # ✅ EXPERIENCE LEVEL
        years = state.years_of_experience or 0
        if years <= 2:
            exp_level = "Beginner"
        elif years <= 6:
            exp_level = "Intermediate"
        else:
            exp_level = "Advanced"

        # ✅ BUILD CONTEXT
        resume_inventory_text = json.dumps({
            "skills": state.user_skills or [],
            "target_role": state.target_role or "Software Engineer",
            "location": state.location or "Remote",
            "github": state.github_username or "",
            "experience_level": exp_level,
            "years_of_experience": years,
            "education": state.education or [],
            "projects": [p.get("name", "") for p in (state.projects or [])],
            "learning_mode": "advanced" if use_existing_skills else "skill_gaps"
        }, indent=2)

        graph_input = {
            "skill_gap_analysis": skill_gap_text,
            "resume_inventory": resume_inventory_text,
        }

        Colors.info("Invoking Curriculum Architect...")
        final_state = cur.app.invoke(graph_input, config={"recursion_limit": 300})

        state.learning_path = (
            final_state.get("draft_path")
            or final_state.get("final_path")
            or final_state
        )

        Colors.success("Learning path generated successfully!")
        state.update_state(SessionState.CURRICULUM_READY)

        #  DISPLAY OUTPUT
        print("\n================ LEARNING PATH ================\n")

        path = state.learning_path

        if hasattr(path, "dict"):
            path = path.dict()

        if isinstance(path, dict) and "path" in path:
            path = path["path"]

        recommendations = []
        if isinstance(path, dict) and "recommendations" in path:
            recommendations = path["recommendations"][:20]
        elif hasattr(path, "recommendations"):
            recommendations = path.recommendations[:20]

        if recommendations:
            for i, r in enumerate(recommendations, 1):
                if isinstance(r, dict):
                    title = r.get('course_title', 'Unknown Course')
                    skill = r.get('skill_name', 'Unknown Skill')
                    url = r.get('course_url', '')
                else:
                    title = getattr(r, 'course_title', 'Unknown Course')
                    skill = getattr(r, 'skill_name', 'Unknown Skill')
                    url = getattr(r, 'course_url', '')

                print(f"{i}. {title} ({skill})")
                if url:
                    print(f"     URL: {url}")
                print()
        else:
            print("⚠ WARNING: Learning path missing recommendations.\n")

        print("=========================================================\n")

        return True

    except Exception as e:
        traceback.print_exc()
        state.add_error(f"Curriculum generation failed: {e}")
        return False
    

def handle_no_skill_gaps(state: GlobalState) -> bool:
    """Handle scenario where no skill gaps are detected.
    Offers user options for courses on existing skills or interview prep.
    
    Returns:
        True if user chose an option, False if declined everything
    """
    Colors.section("No Skill Gaps Detected")
    print("\nGreat news! Based on the job market analysis, you already have")
    print("the skills needed for your target role.")
    print("\nWould you like to:")
    print("  1. Get advanced courses for your existing skills")
    print("  2. Start interview preparation")
    print("  3. Skip for now")
    
    while True:
        choice = input("\nSelect (1/2/3): ").strip()
        
        if choice == '1':
            Colors.info("Generating advanced learning path for your existing skills...")
            return curriculum_wrapper(state, use_existing_skills=True)
        
        elif choice == '2':
            confirm = input("\nWould you like to start a mock interview? (yes/no): ").strip().lower()
            if confirm in ['yes', 'y']:
                return interview_wrapper(state)
            else:
                Colors.info("Interview skipped.")
                return False
        
        elif choice == '3':
            Colors.info("Skipping additional options.")
            return False
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def classify_intent(user_input: str, state: GlobalState) -> str:
    prompt = f"""
You are an intent classification system for a career assistant.

Your task is to classify the user's intent into EXACTLY ONE of the following labels:

- interview → user wants mock interview, practice, preparation
- learning → user wants courses, learning path, curriculum, upskilling
- resume → user wants to create, update, or improve resume/CV
- skill_gaps → user asks about missing skills, weaknesses, gaps, what to improve
- report → user wants summary, report, final output
- exit → user wants to quit, stop, end session
- general → anything else
- show_jobs → user wants to see job postings where he can apply

STRICT RULES:
- Output MUST be exactly one label from the list above
- Do NOT explain
- Do NOT add extra words
- Do NOT use punctuation
- If unsure, return "general"

User input: "{user_input}"

Answer:
"""
    try:
        res = llm.invoke(prompt)
        return res.content.strip().lower()
    except:
        return "general"
# Main Loop
def run_supervisor_loop():
    """Main conversational supervisor loop"""


    state = GlobalState()
    state.update_state(SessionState.INITIALIZED, "Supervisor started")

    # Initial setup
    try:
        print("\nHello! I'm your AI Career Advisor. Let's get started.\n")
        
        if not state.target_role:
            state.target_role = input("What job role are you targeting? (e.g., Full Stack Developer): ").strip() or None
            if state.target_role:
                Colors.success(f"Target role: {state.target_role}")
        
        if not state.location:
            state.location = input("Preferred location (city or 'Remote'): ").strip() or None
            if state.location:
                Colors.success(f"Location: {state.location}")

        print("\nHow would you like to provide your background?")
        print("  1. Upload resume (PDF/DOCX/TXT)")
        print("  2. Provide GitHub username")
        print("  3. Skip for now")
        
        choice = input("\nSelect (1/2/3): ").strip()
        
        if choice == '1':
            p = input("Full path to resume: ").strip().strip('"\'')
            if p and Path(p).exists():
                state.resume_path = p
                Colors.success(f"Using resume: {p}")
                extract_skills_wrapper(state)
            else:
                Colors.error("Path not found. Skipping.")
        elif choice == '2':
            gh = input("GitHub username or URL: ").strip()
            if gh:
                if gh.startswith("https://github.com/"):
                    gh = gh.rstrip('/').split('/')[-1]
                state.github_username = gh
                Colors.success(f"GitHub username: {gh}")
                extract_skills_wrapper(state)
                extract_projects_wrapper(state)
        else:
            Colors.info("Proceeding without resume/GitHub for now.")
        
        state.update_state(SessionState.INPUT_COLLECTED, "User input collected")
        
        # Run initial job analysis if we have the basics
        if state.target_role and state.user_skills:
            Colors.info("\nRunning initial job market analysis...")
            job_market_analysis_wrapper(state)
            state.skill_gaps = state.skill_gaps[:15]
            
            # Handle no skill gaps scenario
            if not state.skill_gaps:
                handle_no_skill_gaps(state)

    except KeyboardInterrupt:
        print("\n\nSetup interrupted. Exiting.")
        return

    # Main conversation loop
    Colors.header("AGENTIC CAREER ADVISOR - Interactive Terminal")
    Colors.info("Talk naturally with the assistant. Type 'exit' to quit.\n")
    Colors.info("Example commands:")
    print("  - 'Find jobs for backend developer in Bangalore'")
    print("  - 'Generate my resume'")
    print("  - 'Start a mock interview'")
    print("  - 'Create learning path'\n")
    
    while True:
        try:
            user_input = input("\nYOU: ").strip()
            if not user_input:
                continue

            
            intent = classify_intent(user_input, state).strip().lower()

# Exit
            if intent == "exit":
                print("\nGoodbye! Saving your session...")
                state.update_state(SessionState.CLEANED_UP, "User exited")
                break

# Interview
            elif intent == "interview":
                Colors.info("Starting interview session...")
                if not state.skill_gaps and state.user_skills:
                    Colors.info("No skill gaps detected. Interview will focus on general competencies.")
                elif not state.skill_gaps:
                    Colors.info("Running job analysis first...")
                    job_market_analysis_wrapper(state)
                interview_wrapper(state)
                continue



            elif intent == "show_jobs":
                if not state.job_data:
                    Colors.info("No jobs available. Run job search first.")
                else:
                    print("\nJobs Found:\n")
                    for i, job in enumerate(state.job_data[:10], 1):
                        print(f"{i}. {job.get('title')} at {job.get('company')}")
                        print(f"   Location: {job.get('location')}")
                        print(f"   Skills: {', '.join(job.get('skills_required', [])[:5])}")
                        print(f"   Link: {job.get('link')}\n")
                continue


            elif intent == "learning":

                
                if state.learning_path:
                    Colors.info("Showing existing learning path...")

                else:
                    
                    if not state.skill_gaps:
                        if not state.user_skills:
                            Colors.info("Extracting skills first...")
                            if not extract_skills_wrapper(state):
                                Colors.error("Cannot build learning path.")
                                continue
                        
                        curriculum_wrapper(state, use_existing_skills=True)
                    else:
                        curriculum_wrapper(state, use_existing_skills=False)

 
                continue
# Resume
            elif intent == "resume":
                Colors.info("Generating resume...")
                generate_resume_wrapper(state, force=True)
                continue

# Skills
            
# Report
            elif intent == "report":
                assemble_and_save_report(state)
                continue

            elif intent == "skill_gaps":
                if not state.skill_gaps:
                    Colors.info("No skill gaps found. Running job analysis...")
                    job_market_analysis_wrapper(state)

                if state.skill_gaps:
                    print("\nYour Skill Gaps:")
                    for i, g in enumerate(state.skill_gaps, 1):
                        print(f"  {i}. {g}")
                else:
                    Colors.info("No skill gaps detected. You're well aligned with the role!")

                continue

# ✅ REAL conversational fallback
            else:
                prompt = f"""
You are a helpful AI career advisor.

User: {user_input}

Context:
- Role: {state.target_role}
- Skills: {state.user_skills}
- Skill gaps: {state.skill_gaps}

Respond naturally and helpfully.
"""
                res = llm.invoke(prompt)
                print(f"\nAI: {res.content}")
                continue
            # Exit command
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
            break
        except Exception as e:
            Colors.error(f"Unexpected error: {e}")
            traceback.print_exc()
            Colors.info("You can continue or type 'exit' to quit.")

    # Save final report on exit
    try:
        fname = assemble_and_save_report(state)
        Colors.success(f"\nFinal report saved: {fname}")
        print(f"\nSession complete! Duration: {int(time.time() - state.start_time)}s")
    except Exception as e:
        Colors.error(f"Failed to save final report: {e}")


if __name__ == "__main__":
    run_supervisor_loop()
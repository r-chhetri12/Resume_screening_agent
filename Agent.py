import io
import html
from typing import TypedDict, Literal, NotRequired

import streamlit as st
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from pypdf import PdfReader
from docx import Document


# ============================================================
# ResumeFit AI — evaluation pipeline + Streamlit interface
# ============================================================
st.set_page_config(
    page_title="ResumeFit AI | Career Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root {--ink:#e8edf7;--muted:#9aa8bd;--line:rgba(148,163,184,.18);}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 12% 0%,rgba(50,91,155,.20),transparent 38%),#080f1b;color:var(--ink)}
[data-testid="stHeader"]{background:rgba(8,15,27,.85)}
[data-testid="stSidebar"]{background:#0b1422;border-right:1px solid var(--line)}
.block-container{max-width:1220px;padding-top:2rem;padding-bottom:4rem}
h1,h2,h3{font-family:'Manrope',sans-serif!important;letter-spacing:-.035em}
.brand{font-family:'Manrope',sans-serif;font-weight:800;font-size:1.28rem;color:#f3f6fc}
.brand-mark{display:inline-flex;width:34px;height:34px;align-items:center;justify-content:center;border-radius:11px;background:linear-gradient(135deg,#6ea4ff,#8b7cff);color:#07101d;font-weight:900;margin-right:9px}
.hero{border:1px solid var(--line);border-radius:24px;padding:2.2rem 2.3rem;background:linear-gradient(115deg,rgba(29,55,91,.75),rgba(17,27,43,.82));box-shadow:0 18px 60px rgba(0,0,0,.18);margin-bottom:1.6rem}
.eyebrow,.section-kicker{color:#8db7ff;text-transform:uppercase;letter-spacing:.15em;font-size:.72rem;font-weight:700}
.hero h1{font-size:clamp(2rem,4vw,3.1rem);margin:.55rem 0;color:#f4f7fc}
.hero p,.subtle{color:#aebbd0;line-height:1.65}
.upload-card,.mini-card{background:rgba(17,27,43,.85);border:1px solid var(--line);border-radius:17px;padding:1rem 1.1rem;height:100%}
.upload-title,.mini-card h4{font-family:'Manrope',sans-serif;font-weight:700;color:#edf3ff;font-size:1.02rem}
.result-panel{background:linear-gradient(135deg,rgba(23,42,67,.95),rgba(15,25,40,.96));border:1px solid var(--line);border-radius:20px;padding:1.4rem 1.5rem}
.result-label{color:#9fb0c8;text-transform:uppercase;font-size:.7rem;letter-spacing:.13em;font-weight:700}
.result-good{color:#83e0b1;font-family:'Manrope',sans-serif;font-size:1.8rem;font-weight:800}
.result-warn{color:#ffd18a;font-family:'Manrope',sans-serif;font-size:1.8rem;font-weight:800}
.stButton>button,.stDownloadButton>button{border-radius:12px;min-height:2.9rem;font-weight:700}
.stButton>button[kind="primary"]{background:linear-gradient(100deg,#6d9fff,#8a83ff);color:#07101d;border:0}
div[data-testid="stFileUploader"]{background:rgba(8,15,27,.4);border:1px dashed rgba(120,169,255,.35);border-radius:13px;padding:.45rem}
div[data-testid="stTextArea"] textarea{background:#0c1727;color:#e8edf7;border:1px solid var(--line);border-radius:12px}
.stTabs [data-baseweb="tab-list"]{gap:8px;border-bottom:1px solid var(--line)}
.stTabs [aria-selected="true"]{color:#8db7ff!important}
[data-testid="stMetric"]{background:#111b2b;border:1px solid var(--line);padding:1rem;border-radius:14px}
hr{border-color:var(--line)} footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "qwen2.5:3b"
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TEXT_CHARS = 24000


# -----------------------------
# Structured output schemas
# Keep fields simple and explicit for local 3B models.
# -----------------------------
class ScreeningSchema(BaseModel):
    recommendation: Literal["Good_to_apply", "Need_improvement"] = Field(
        description="Overall recommendation based on essential job requirements, not preferred extras"
    )
    fit_level: Literal["strong", "moderate", "weak"] = Field(
        description="Overall evidence alignment: strong, moderate, or weak"
    )
    fit_score: int = Field(ge=0, le=10, description="Resume-to-job-description evidence fit score from 0 to 10")
    selection_outlook: Literal["Promising", "Competitive", "Stretch"] = Field(
        description="Qualitative resume-based shortlist outlook, not a hiring probability"
    )
    selection_outlook_reason: str = Field(description="Brief explanation of the outlook and its uncertainty")
    resume_improvements: list[str] = Field(
        description="Three to five prioritized, job-specific resume edits supported by true candidate evidence"
    )
    reason: str = Field(description="Evidence-based explanation in 2 to 4 sentences")
    matched_required_skills: list[str] = Field(
        description="Mandatory skills clearly supported by the resume, including project evidence"
    )
    partially_matched_skills: list[str] = Field(
        description="Mandatory skills for which the resume gives partial or adjacent evidence"
    )
    missing_or_unverified_required_skills: list[str] = Field(
        description="Mandatory skills not clearly demonstrated; say unverified rather than claiming the candidate cannot do them"
    )
    matched_preferred_skills: list[str] = Field(
        description="Preferred or bonus skills supported by the resume"
    )
    relevant_projects_and_experience: list[str] = Field(
        description="Specific project, internship, work, or academic evidence relevant to the role"
    )
    education_assessment: str = Field(description="How the education requirement aligns with the resume")
    experience_assessment: str = Field(description="Assess experience fairly for the stated seniority")
    critical_gaps: list[str] = Field(
        description="Only gaps that are genuinely essential and could prevent the candidate performing the role"
    )


class DiagnosisSchema(BaseModel):
    skills_to_develop: list[str] = Field(
        description="Specific required skills to develop or demonstrate, not generic advice"
    )
    resume_weaknesses: list[str] = Field(
        description="Evidence-based weaknesses in how the resume presents the candidate"
    )
    resume_edits: list[str] = Field(
        description="Concrete edits to make using only true information already in the resume"
    )
    project_suggestions: list[str] = Field(
        description="Practical project or portfolio work that would close important gaps"
    )
    interview_preparation: list[str] = Field(
        description="Role-specific topics the candidate should prepare to explain"
    )


class ResumeState(TypedDict):
    resume: str
    job_description: str
    screening: NotRequired[dict]
    sentiment: NotRequired[str]
    decision_reason: NotRequired[str]
    matched_skills: NotRequired[list[str]]
    diagnosis: NotRequired[dict]
    response: NotRequired[str]


def _clean_list(values):
    """Normalize model-produced list fields before putting them in graph state."""
    if not isinstance(values, list):
        return []
    cleaned = []
    for value in values:
        item = str(value).strip()
        if item and item not in cleaned:
            cleaned.append(item)
    return cleaned


def _trim_input(text: str, label: str) -> str:
    text = text.strip()
    if len(text) > MAX_TEXT_CHARS:
        st.warning(
            f"{label} is longer than {MAX_TEXT_CHARS:,} characters. "
            f"Only the first {MAX_TEXT_CHARS:,} characters will be analyzed."
        )
        text = text[:MAX_TEXT_CHARS]
    return text


@st.cache_resource
def build_agent():
    # temperature=0 improves repeatability; the detailed rubric below does
    # more to improve consistency than simply increasing model temperature.
    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    screening_llm = llm.with_structured_output(ScreeningSchema)
    diagnosis_llm = llm.with_structured_output(DiagnosisSchema)

    def find_sentiment(state: ResumeState):
        prompt = f"""
ROLE
You are a careful, fair recruiter assessing a candidate for ONE specific job.
Your job is to evaluate evidence, not to reject people for missing keywords.

CANDIDATE RESUME
<resume>
{state["resume"]}
</resume>

JOB DESCRIPTION
<job_description>
{state["job_description"]}
</job_description>

EVALUATION METHOD — FOLLOW IN ORDER
1. Separate the JD's mandatory/required criteria from preferred/bonus criteria.
2. Compare each mandatory criterion with the entire resume. Look in the summary,
   education, internship, employment, projects, tools, and measurable outcomes.
3. Count a skill as matched when there is credible evidence. A project can prove
   practical skill; professional employment is NOT required for a fresher role.
4. Use partial match when the resume shows adjacent or introductory evidence.
5. Put a criterion under missing_or_unverified only when it is mandatory and no
   clear evidence appears. This means "not demonstrated in this resume", NOT
   "the candidate does not know it".
6. Preferred skills must not be treated as mandatory or used as critical gaps.
7. Respect the experience level. For 0–2 years / fresher roles, relevant projects,
   coursework and internships can satisfy experience expectations.
8. Do not invent tools, years, certifications, metrics, responsibilities or results.
9. If the JD is vague or has no explicit required skills, use responsibilities and
   role expectations; do not fabricate a long list of requirements.

DECISION CALIBRATION
- Good_to_apply: the candidate meets most essential requirements, has relevant
  evidence, and has no major essential gap. Some unverified skills are acceptable.
- Need_improvement: one or more genuinely essential requirements are missing,
  or the evidence is too limited for this role.
- Do NOT choose Need_improvement merely because preferred skills are absent.
- fit_level is strong/moderate/weak based on the whole profile, not keyword count.
- A moderate but credible entry-level fit can still be Good_to_apply.

FIT SCORE (0–10): This is resume/JD evidence alignment, NOT selection probability.
0–2 = very little relevant evidence; 3–4 = weak alignment; 5–6 = partial/moderate
alignment with meaningful gaps; 7–8 = good alignment with most essential criteria
supported; 9–10 = exceptionally strong evidence across nearly all essential criteria.
Weight essential requirements and evidence quality more than keyword count. For
fresher roles, relevant projects and internships count. Do not over-penalize missing
preferred skills and do not inflate the score.

SHORTLIST OUTLOOK:
Promising = strong essential-requirement alignment.
Competitive = reasonable alignment but meaningful gaps/uncertainty.
Stretch = weak alignment or important essential requirements not demonstrated.
This is a qualitative resume-only outlook, NOT a percentage chance or guarantee.
Actual selection depends on the applicant pool, interviews, employer criteria, timing
and other factors.

RESUME IMPROVEMENTS: Return 3–5 prioritized, job-specific edits. State where to make
each change (summary, skills, project bullet, education, etc.). Use only facts in the
resume; never invent metrics, skills, responsibilities or experience.

OUTPUT RULES
- Be specific and balanced.
- matched_required_skills: only evidence-supported mandatory skills.
- partially_matched_skills: partial/adjacent mandatory evidence.
- missing_or_unverified_required_skills: mandatory only.
- matched_preferred_skills: preferred skills with evidence.
- relevant_projects_and_experience: name the evidence and explain its relevance.
- critical_gaps: only major mandatory gaps, not every missing keyword.
- reason: 2–4 sentences, explain the decision using resume and JD evidence.
"""
        result = screening_llm.invoke(prompt)
        data = result.model_dump()
        data["matched_required_skills"] = _clean_list(data.get("matched_required_skills"))
        data["partially_matched_skills"] = _clean_list(data.get("partially_matched_skills"))
        data["missing_or_unverified_required_skills"] = _clean_list(
            data.get("missing_or_unverified_required_skills")
        )
        data["matched_preferred_skills"] = _clean_list(data.get("matched_preferred_skills"))
        data["relevant_projects_and_experience"] = _clean_list(
            data.get("relevant_projects_and_experience")
        )
        data["critical_gaps"] = _clean_list(data.get("critical_gaps"))
        data["resume_improvements"] = _clean_list(data.get("resume_improvements"))
        try:
            data["fit_score"] = max(0, min(10, int(data.get("fit_score", 0))))
        except (TypeError, ValueError):
            data["fit_score"] = 0

        # Guard against a common small-model failure: treating preferred skills
        # or unverified resume keywords as automatic disqualifiers.
        data["recommendation"] = (
            "Need_improvement"
            if data["critical_gaps"]
            else data["recommendation"]
        )
        return {
            "screening": data,
            "sentiment": data["recommendation"],
            "decision_reason": data["reason"],
            "matched_skills": data["matched_required_skills"],
        }

    def check_sentiment(state: ResumeState) -> Literal["good_to_apply", "run_diagnosis"]:
        if state.get("sentiment") == "Good_to_apply":
            return "good_to_apply"
        return "run_diagnosis"

    def good_to_apply(state: ResumeState):
        # Use the screening evidence already produced. Avoid a second LLM call
        # that can contradict the first call or invent new candidate details.
        screening = state.get("screening", {})
        strengths = screening.get("relevant_projects_and_experience", [])
        partial = screening.get("partially_matched_skills", [])
        preferred = screening.get("matched_preferred_skills", [])
        lines = [
            "Your resume appears suitable for applying to this role.",
            "",
            "Why your profile aligns:",
        ]
        lines.extend(f"- {x}" for x in strengths[:6])
        if not strengths:
            lines.append("- Your resume demonstrates relevant alignment with the role requirements.")
        if partial:
            lines += ["", "A few areas to review before applying:"]
            lines.extend(f"- {x}" for x in partial[:5])
        if preferred:
            lines += ["", "Relevant preferred skills also demonstrated:"]
            lines.extend(f"- {x}" for x in preferred[:5])
        lines += [
            "",
            "Before applying, tailor the resume summary to this role, make project outcomes "
            "and tools easy to find, and verify that every claim is accurate.",
            "This is an evidence-based screening aid, not a guarantee of interview selection.",
        ]
        return {"response": "\n".join(lines)}

    def run_diagnosis(state: ResumeState):
        screening = state.get("screening", {})
        prompt = f"""
You are an evidence-based career coach. Give a focused improvement diagnosis
for this candidate and this exact job. Do not repeat generic resume advice.

RESUME:
{state["resume"]}

JOB DESCRIPTION:
{state["job_description"]}

SCREENING FINDINGS:
Reason: {screening.get("reason", "")}
Matched mandatory skills: {screening.get("matched_required_skills", [])}
Partially matched skills: {screening.get("partially_matched_skills", [])}
Missing/unverified mandatory skills: {screening.get("missing_or_unverified_required_skills", [])}
Matched preferred skills: {screening.get("matched_preferred_skills", [])}
Relevant project/experience evidence: {screening.get("relevant_projects_and_experience", [])}
Critical gaps: {screening.get("critical_gaps", [])}

INSTRUCTIONS:
- Prioritize mandatory gaps. Do not recommend learning optional skills before essential ones.
- Distinguish resume presentation problems from genuine skill-development needs.
- Suggest only edits supported by facts already in the resume; never fabricate experience.
- Suggest a project only when it would provide evidence for a real JD requirement.
- Make recommendations specific, practical, and suitable for the stated seniority.
- If a skill is simply not mentioned, say "demonstrate or verify this skill", not "you do not know it".
- Return concise, useful lists. Avoid vague advice such as "improve communication" unless the JD requires it.
"""
        result = diagnosis_llm.invoke(prompt)
        return {"diagnosis": result.model_dump()}

    def improve_skill(state: ResumeState):
        # Render the structured diagnosis directly rather than asking the
        # small model for a third free-form rewrite that may drift.
        diagnosis = state.get("diagnosis", {})
        screening = state.get("screening", {})
        sections = [
            "Overall assessment",
            screening.get("reason", "The resume has some gaps against the target role."),
            "",
            "Skills to develop or demonstrate",
        ]
        sections.extend(f"- {x}" for x in _clean_list(diagnosis.get("skills_to_develop")))
        sections += ["", "Resume weaknesses"]
        sections.extend(f"- {x}" for x in _clean_list(diagnosis.get("resume_weaknesses")))
        sections += ["", "Recommended resume edits"]
        sections.extend(f"- {x}" for x in _clean_list(diagnosis.get("resume_edits")))
        sections += ["", "Portfolio / project suggestions"]
        sections.extend(f"- {x}" for x in _clean_list(diagnosis.get("project_suggestions")))
        sections += ["", "Interview preparation"]
        sections.extend(f"- {x}" for x in _clean_list(diagnosis.get("interview_preparation")))
        sections += [
            "",
            "Important: A skill not shown in a resume is unverified, not proof that "
            "the candidate does not possess it. Only add claims that are true.",
        ]
        return {"response": "\n".join(sections)}

    graph = StateGraph(ResumeState)
    graph.add_node("find_sentiment", find_sentiment)
    graph.add_node("good_to_apply", good_to_apply)
    graph.add_node("run_diagnosis", run_diagnosis)
    graph.add_node("improve_skill", improve_skill)

    graph.add_edge(START, "find_sentiment")
    graph.add_conditional_edges(
        "find_sentiment",
        check_sentiment,
        {
            "good_to_apply": "good_to_apply",
            "run_diagnosis": "run_diagnosis",
        },
    )
    graph.add_edge("good_to_apply", END)
    graph.add_edge("run_diagnosis", "improve_skill")
    graph.add_edge("improve_skill", END)
    return graph.compile()


# -----------------------------
# Document text extraction
# -----------------------------
def extract_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    file_name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("Please upload a file smaller than 10 MB.")

    if file_name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif file_name.endswith(".docx"):
        document = Document(io.BytesIO(raw))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        for table in document.tables:
            for row in table.rows:
                text += "\n" + " | ".join(cell.text for cell in row.cells)
    elif file_name.endswith(".txt"):
        text = raw.decode("utf-8", errors="replace")
    else:
        raise ValueError("Unsupported format. Upload a PDF, DOCX or TXT file.")

    text = text.strip()
    if not text:
        raise ValueError(
            "No readable text was found. Scanned PDFs need OCR first; "
            "upload a text-based PDF, DOCX or TXT file."
        )
    return text


# -----------------------------
# User interface
# -----------------------------
with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">✦</span>ResumeFit AI</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtle" style="margin:8px 0 24px 2px;">CAREER INTELLIGENCE PLATFORM</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**WORKSPACE**")
    st.markdown("◈ &nbsp; Resume analysis")
    st.markdown("◇ &nbsp; Conditional screening workflow")
    st.markdown("---")
    st.markdown("**MODEL STATUS**")
    st.markdown("🟢 &nbsp; Local AI workflow")
    st.caption(f"Ollama model · {MODEL_NAME}")
    st.caption("Documents are processed by your configured local model. Avoid uploading sensitive information to shared deployments.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">✦ &nbsp; AI-powered career intelligence</div>
  <h1>Make your next application<br>your strongest one.</h1>
  <p>Compare your resume with a job description, discover your strengths, and get focused recommendations to move forward with confidence.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-kicker">01 / YOUR DOCUMENTS</div>', unsafe_allow_html=True)
st.markdown("## Start with your resume and target role")
st.markdown('<p class="subtle">Upload your resume, then add the job description you want to be evaluated against.</p>', unsafe_allow_html=True)
left, right = st.columns(2, gap="large")
with left:
    st.markdown('<div class="upload-card"><div class="upload-title">▤ &nbsp; Candidate resume</div><div class="subtle">Your current profile · PDF, DOCX or TXT · up to 10 MB</div></div>', unsafe_allow_html=True)
    resume_file = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"], key="resume_upload", label_visibility="collapsed")
    if resume_file:
        st.caption(f"✓ Attached: {resume_file.name}")
with right:
    st.markdown('<div class="upload-card"><div class="upload-title">▤ &nbsp; Target job description</div><div class="subtle">Upload a file or paste the role details</div></div>', unsafe_allow_html=True)
    jd_file = st.file_uploader("Upload job description", type=["pdf", "docx", "txt"], key="jd_upload", label_visibility="collapsed")
    if jd_file:
        st.caption(f"✓ Attached: {jd_file.name}")

with st.expander("Or paste the job description", expanded=not bool(jd_file)):
    jd_text = st.text_area(
        "Job description", height=190,
        placeholder="Paste the role summary, responsibilities, required skills and qualifications here...",
        label_visibility="collapsed",
    )

st.markdown("<br>", unsafe_allow_html=True)
run_clicked = st.button("✦  Analyze resume against this role", type="primary", use_container_width=True)

if run_clicked:
    try:
        resume_text = extract_text(resume_file)
        uploaded_jd_text = extract_text(jd_file) if jd_file else ""
        job_description = jd_text.strip() or uploaded_jd_text
        if not resume_text:
            st.error("Please upload your resume before starting the analysis.")
            st.stop()
        if not job_description:
            st.error("Please upload or paste a job description.")
            st.stop()
        resume_text = _trim_input(resume_text, "Resume")
        job_description = _trim_input(job_description, "Job description")
        if len(resume_text) < 100:
            st.warning("The extracted resume text is unusually short. Check that the file contains selectable text.")

        with st.status("Screening your resume...", expanded=True) as status_box:
            st.write("Extracting resume and job requirements")
            st.write("Evaluating required skills separately from preferred skills")
            st.write("Checking projects, education and experience evidence")
            st.write("Preparing the recommendation and improvement path")
            result = build_agent().invoke(
                {"resume": resume_text, "job_description": job_description}
            )
            status_box.update(label="Analysis complete", state="complete", expanded=False)

        st.session_state["resume_result"] = result
        st.success("Your resume analysis is ready.")
    except Exception as exc:
        st.error(f"The analysis could not be completed: {exc}")
        st.info(
            "Check that Ollama is running and that qwen2.5:3b is installed. "
            "If structured-output parsing fails, update langchain-ollama and "
            "langchain-core, then restart Streamlit."
        )

result = st.session_state.get("resume_result")
if result:
    screening = result.get("screening") or {}
    diagnosis = result.get("diagnosis") or {}
    status = result.get("sentiment", "Unknown")
    reason = result.get("decision_reason", "")
    matched = _clean_list(result.get("matched_skills", []))
    partial = _clean_list(screening.get("partially_matched_skills", []))
    missing = _clean_list(screening.get("missing_or_unverified_required_skills", []))
    preferred = _clean_list(screening.get("matched_preferred_skills", []))
    headline = "Good to apply" if status == "Good_to_apply" else "Needs improvement"
    css_class = "result-good" if status == "Good_to_apply" else "result-warn"
    status_note = (
        "Your resume appears to meet the core requirements for this role."
        if status == "Good_to_apply"
        else "There are important requirements or evidence gaps worth addressing."
    )

    st.markdown("---")
    st.markdown('<div class="section-kicker">02 / SCREENING OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown("## Your results")
    st.markdown(
        f'<div class="result-panel"><div class="result-label">AI screening recommendation</div>'
        f'<div class="{css_class}">{html.escape(headline)}</div>'
        f'<div style="color:#b3c0d4;margin-top:5px;">{html.escape(status_note)}</div>'
        f'<hr style="border-color:rgba(148,163,184,.18);">'
        f'<div style="color:#e0e8f5;font-weight:600;margin-bottom:5px;">Why this result?</div>'
        f'<div style="color:#aebbd0;line-height:1.6;">{html.escape(reason)}</div></div>',
        unsafe_allow_html=True,
    )

    score, outlook_col = st.columns([1, 2])
    score.metric("Resume–JD fit score", f"{screening.get('fit_score', 0)}/10")
    with outlook_col:
        st.markdown(f"**Shortlist outlook: {screening.get('selection_outlook', 'Not assessed')}**")
        st.write(screening.get("selection_outlook_reason", "No explanation returned."))
    st.progress(max(0, min(10, int(screening.get("fit_score", 0)))) / 10)
    st.caption(
        "This score measures resume-to-job-description alignment. It is NOT a percentage "
        "chance of being shortlisted or selected."
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Matched required skills", len(matched))
    m2.metric("Partially matched", len(partial))
    m3.metric("Unverified required", len(missing))

    tab1, tab2, tab3, tab4 = st.tabs([
        "✦ Skill alignment", "↗ Resume changes", "▤ Improvement plan", "◎ Final recommendation"
    ])
    with tab1:
        st.markdown("### Required skills demonstrated")
        if matched:
            for item in matched:
                st.markdown(f"- {item}")
        else:
            st.info("No mandatory skills were clearly identified as matched.")
        if partial:
            st.markdown("### Partially demonstrated")
            for item in partial:
                st.markdown(f"- {item}")
        if missing:
            st.markdown("### Required skills not demonstrated in the resume")
            for item in missing:
                st.markdown(f"- {item}")
            st.caption("Not demonstrated does not mean you do not possess the skill.")
        if preferred:
            st.markdown("### Preferred skills demonstrated")
            for item in preferred:
                st.markdown(f"- {item}")
        evidence = _clean_list(screening.get("relevant_projects_and_experience", []))
        if evidence:
            st.markdown("### Relevant project and experience evidence")
            for item in evidence:
                st.markdown(f"- {item}")
        st.markdown("### Education")
        st.write(screening.get("education_assessment", "Not assessed"))
        st.markdown("### Experience")
        st.write(screening.get("experience_assessment", "Not assessed"))

    with tab2:
        st.markdown("### Recommended changes for this specific job")
        edits = _clean_list(screening.get("resume_improvements", []))
        if edits:
            for i, item in enumerate(edits, start=1):
                st.markdown(f"**{i}.** {item}")
        else:
            st.info("No specific edits were returned. Use the skill alignment section to tailor your resume.")
        st.caption("Only add claims that are accurate and can be supported with real examples.")

    with tab3:
        if diagnosis:
            groups = [
                ("Skills to develop or demonstrate", "skills_to_develop"),
                ("Resume weaknesses", "resume_weaknesses"),
                ("Recommended resume edits", "resume_edits"),
                ("Portfolio / project suggestions", "project_suggestions"),
                ("Interview preparation", "interview_preparation"),
            ]
            for heading, key in groups:
                st.markdown(f"#### {heading}")
                values = _clean_list(diagnosis.get(key, []))
                if values:
                    for item in values:
                        st.markdown(f"- {item}")
                else:
                    st.caption("No specific items returned.")
        elif status == "Good_to_apply":
            st.success("No separate gap diagnosis was required by the conditional workflow.")
            if partial:
                st.markdown("#### Small areas to review")
                for item in partial:
                    st.markdown(f"- {item}")
        else:
            st.info("No diagnosis details were returned.")

    with tab4:
        st.markdown("### Personalized recommendation")
        st.markdown(result.get("response", "No recommendation was generated."))
        if screening.get("critical_gaps"):
            st.markdown("### Critical gaps")
            for item in _clean_list(screening.get("critical_gaps", [])):
                st.markdown(f"- {item}")

    report_parts = [
        f"ResumeFit AI result: {headline}",
        f"Reason: {reason}",
        f"Fit level: {screening.get('fit_level', 'Not provided')}",
        f"Resume-JD fit score: {screening.get('fit_score', 'Not provided')}/10",
        f"Shortlist outlook: {screening.get('selection_outlook', 'Not provided')}",
        f"Outlook explanation: {screening.get('selection_outlook_reason', '')}",
        "\\nRecommended resume changes:\\n" + "\\n".join(
            f"- {x}" for x in _clean_list(screening.get("resume_improvements", []))
        ),
        "\nMatched required skills:\n" + "\n".join(f"- {x}" for x in matched),
        "\nPartially matched skills:\n" + "\n".join(f"- {x}" for x in partial),
        "\nMissing or unverified required skills:\n" + "\n".join(f"- {x}" for x in missing),
        "\nMatched preferred skills:\n" + "\n".join(f"- {x}" for x in preferred),
        "\nRelevant projects and experience:\n" + "\n".join(
            f"- {x}" for x in _clean_list(screening.get("relevant_projects_and_experience", []))
        ),
        "\nEducation assessment:\n" + screening.get("education_assessment", ""),
        "\nExperience assessment:\n" + screening.get("experience_assessment", ""),
        "\nCritical gaps:\n" + "\n".join(f"- {x}" for x in _clean_list(screening.get("critical_gaps", []))),
    ]
    for key, heading in [
        ("skills_to_develop", "Skills to develop"),
        ("resume_weaknesses", "Resume weaknesses"),
        ("resume_edits", "Resume edits"),
        ("project_suggestions", "Project suggestions"),
        ("interview_preparation", "Interview preparation"),
    ]:
        report_parts.append(
            f"\n{heading}:\n" + "\n".join(f"- {x}" for x in _clean_list(diagnosis.get(key, [])))
        )
    report_parts.append("\nRecommendation:\n" + result.get("response", ""))
    st.download_button(
        "↓  Download your analysis report",
        data="\n".join(report_parts),
        file_name="resumefit_analysis.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.caption(
        "ResumeFit AI is a decision-support tool, not an automated hiring decision. "
        "The fit score is not a selection probability. Actual outcomes depend on the "
        "employer, applicant pool, interview and other factors. A skill not shown in a "
        "resume may still be one you possess."
    )
else:
    st.markdown("---")
    st.markdown('<div class="section-kicker">03 / WHAT YOU WILL RECEIVE</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3, gap="medium")
    with a:
        st.markdown('<div class="mini-card"><h4>✦ Skill alignment</h4><div class="subtle">Separate required and preferred skills, with evidence from your resume.</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="mini-card"><h4>↗ Actionable feedback</h4><div class="subtle">Get specific recommendations for your skills, projects and resume sections.</div></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="mini-card"><h4>▤ Clear recommendation</h4><div class="subtle">Receive a balanced recommendation with a plain-language explanation.</div></div>', unsafe_allow_html=True)

ResumeFit AI
An AI-powered resume screening and improvement assistant built with
Streamlit, LangGraph, LangChain, Pydantic, and Ollama.

ResumeFit AI compares a candidate's resume with a specific job
description (JD), identifies evidence of role alignment, and provides
practical recommendations to improve the resume for that role.
Important: ResumeFit AI is a decision-support tool. Its fit score
is not a probability of getting shortlisted or hired, and its
recommendation is not an actual hiring decision.

Features
- Resume and JD input: Upload a resume and a job description, or
  paste the JD into the app.
- Document formats: PDF, DOCX, and TXT.
- Structured screening: Separates required skills from preferred
  skills and checks evidence across projects, education, internships,
  and experience.
- Fit score out of 10: Estimates how closely the resume
  demonstrates the requirements in the supplied JD.
- Qualitative shortlist outlook: Returns Promising,
  Competitive, or Stretch, with an explanation.
- Skill alignment: Lists matched, partially matched, and required
  skills that are not demonstrated in the resume.
- Conditional LangGraph workflow: Routes the result through either
  a positive recommendation path or a deeper improvement-diagnosis
  path.
- Resume improvement suggestions: Recommends specific, job-focused
  changes to relevant resume sections.
- Development guidance: Suggests skills to develop, portfolio
  projects to consider, and interview topics to prepare.
- Downloadable report: Export screening findings and
  recommendations as a text report.
Tech stack
  Technology    Purpose
  Python        Application logic
Streamlit     Web interface
LangGraph     Conditional screening workflow
LangChain     Model integration
Ollama        Runs the language model locally
Qwen 2.5 3B   Default local language model
Pydantic      Structured model output and validation
pypdf         PDF text extraction
python-docx   DOCX text extraction
How the workflow works
Resume + Job Description
          |
          v
   Extract document text
          |
          v
  Screen resume against JD
          |
          v
 Evaluate required and preferred
       skills separately
          |
          v
   Conditional LangGraph route
       /               \
      /                 \
Good to apply       Needs improvement
      |                 |
      |           Run diagnosis
      |                 |
      |           Build improvement
      |              plan
      \                 /
       \               /
          v
   Display results and
   downloadable report
The screening stage produces structured findings. If the candidate is
assessed as needing improvement, the graph runs a second structured
diagnosis step. If the candidate is assessed as suitable to apply, the
app uses the existing screening evidence to prepare the recommendation.
Fit score guide
The score is an evidence-based resume-to-JD alignment rating, not a
hiring probability.
  Score                               Interpretation
  0--2                                Very little relevant evidence

  3--4                                Weak alignment; several essential
requirements are unsupported

  5--6                                Partial or moderate alignment;
meaningful gaps remain

  7--8                                Good alignment; most essential
requirements have supporting
evidence
9--10                               Very strong alignment; nearly all
essential requirements have clear
evidence
The model should consider the quality and importance of evidence, not
just keyword frequency. Relevant academic projects and internships can
count as evidence for fresher roles.
Shortlist outlook
- Promising: Strong alignment with the stated essential
  requirements.
- Competitive: Reasonable alignment, but some gaps or uncertainty
  remain.
- Stretch: Weak alignment or important essential requirements are
  not demonstrated.
These labels are qualitative guidance only. Real hiring outcomes depend
on the employer's criteria, applicant pool, interview performance,
communication, timing, and other factors.
Requirements
- Python 3.10 or newer
- Ollama installed on the machine running the app
- The qwen2.5:3b model downloaded in Ollama
- Internet access may be needed to install Python packages. Model
  inference is configured to use the local Ollama service.
Installation and setup
1. Clone the repository
git clone <your-github-repository-url>
cd <your-repository-folder>
Replace the placeholders with your actual GitHub repository URL and
folder name.
2. Create and activate a virtual environment
Windows PowerShell:
python -m venv .venv
.venv\Scripts\Activate.ps1
macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
4. Install and start Ollama
Install Ollama from https://ollama.com.

Pull the configured model:
ollama pull qwen2.5:3b
Make sure Ollama is running. In a separate terminal, you can check the
model with:
ollama run qwen2.5:3b
Exit the model prompt after confirming it responds.
5. Run the Streamlit app
From the project directory:
streamlit run app.py
Open the local address printed by Streamlit, commonly
http://localhost:8501.
How to use ResumeFit AI
1. Start the application.
2. Upload your resume as a PDF, DOCX, or TXT file.
3. Upload the job description or paste its text into the JD field.
4. Select Analyze resume against this role.
5. Review the fit score and qualitative shortlist outlook.
6. Check the skill-alignment section for matched, partially matched,
   and unverified requirements.
7. Review the job-specific resume changes and improvement plan.
8. Download the analysis report if needed.
For the best results, provide a complete and readable JD containing role
responsibilities, essential qualifications, and preferred skills.
Project structure
ResumeFit_AI/
├── app.py
├── requirements.txt
└── README.md
File handling and limitations
- Maximum supported upload size is 10 MB per file.
- Supported formats are PDF, DOCX, and TXT.
- PDF extraction reads selectable text. Scanned/image-only PDFs are
  not OCR-processed; convert them to searchable PDFs or use OCR
  before uploading.
- The model can misunderstand ambiguous job descriptions or resume
  wording. Review the output rather than treating it as definitive.
- A skill missing from the resume is treated as not demonstrated in
  the document, not proof that the candidate does not have that
  skill.
- Only add resume claims, skills, metrics, or experience that are true
  and that you can explain in an interview.
- The default model is qwen2.5:3b. Results can vary with model
  capability, prompt interpretation, and document quality.
Deployment note
The current configuration uses a local Ollama model. A hosted Streamlit
application cannot automatically access Ollama running on your personal
laptop. To deploy this version for public access, host the model/API in
an appropriately secured environment or adapt the application to use a
hosted model provider.

Do not expose an unauthenticated Ollama endpoint directly to the
public internet. Protect model endpoints with authentication, network
controls, and HTTPS.
Responsible use
ResumeFit AI is intended to help candidates understand how clearly their
resumes demonstrate a particular job's requirements. It should not be
used as the sole basis for hiring, rejection, or other high-impact
employment decisions. Review outputs for accuracy, fairness, and
context.
Future improvements
- Add OCR support for scanned resumes.
- Add tests for document extraction, scoring boundaries, and graph
  routing.
- Add evaluation datasets to measure screening consistency and reduce
  unsupported claims.
- Add configurable model/provider settings.
- Add secure deployment configuration and authentication.
- Add PDF report export and resume section-by-section feedback.
License
Add a license before distributing this project. If you intend to publish
it as an open-source project, choose a license that matches how you want
others to use, modify, and distribute it.

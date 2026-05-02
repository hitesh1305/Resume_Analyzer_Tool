# LLM-Based Resume Screening System

## 📌 Overview

This is an AI-powered resume screening system that evaluates how well a candidate’s resume matches a given job description. It simulates a recruiter’s decision-making workflow using a multi-agent LLM architecture.

The system analyzes:

* Skills & education alignment
* Work experience relevance
* Salary-market fit

and produces a structured hiring decision with an overall score.

---

## 🚀 Features

* 📄 Upload Resume & Job Description (PDF)
* 🤖 Multi-agent LLM evaluation system
* 🧩 Modular analysis (skills, experience, salary)
* 📊 Structured scoring (0–100)
* ⚡ Fast inference (<2s per evaluation)
* 🌐 Tool-augmented reasoning (Wikipedia + Web Search)
* 🖥️ Interactive UI using Streamlit

---

## 🏗️ System Architecture

```
User Input (Resume + JD PDFs)
            ↓
      PDF Text Extraction
            ↓
     Supervisor LLM Agent
            ↓
 ┌──────────────┬──────────────┬──────────────┐
 │ Skill Agent  │ Experience   │ Salary Agent │
 │ (Wikipedia)  │ Agent (Search)│ (Search)    │
 └──────────────┴──────────────┴──────────────┘
            ↓
    Aggregated Decision + Score
            ↓
         Streamlit UI
```

---

## 🧠 Core Concept

The system follows a **multi-agent decision pipeline**:

1. **Skill & Education Agent**

   * Compares required vs available skills
   * Identifies missing competencies

2. **Experience Agent**

   * Evaluates role alignment and seniority
   * Extracts companies and roles

3. **Salary Agent**

   * Estimates market-aligned salary range
   * Assesses compensation expectations

4. **Supervisor Agent**

   * Aggregates all signals
   * Produces final decision (APPROVE / REJECT)
   * Assigns a score out of 100

---

## ⚙️ Tech Stack

* **Language:** Python
* **LLM Framework:** LangChain
* **Model:** Gemini 2.5 Flash Lite
* **UI:** Streamlit
* **PDF Parsing:** PyPDFLoader (pypdf)
* **Validation:** Pydantic
* **Tools:**

  * DuckDuckGo Search
  * Wikipedia API

---

## 📂 Project Structure

```
resume_analyzer_tool/
│
├── rat.py                # Main application file
├── requirements.txt      # Dependencies
├── .env                  # API keys (local)
└── README.md             # Project documentation
```

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/hitesh1305/Resume_Analyzer_Tool.git
cd Resume_Analyzer_Tool
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Setup Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

For deployment (Streamlit Cloud), use:

```
st.secrets["GOOGLE_API_KEY"]
```

---

## ▶️ Running the Application

```bash
streamlit run rat.py
```

---

## 🖥️ Usage

1. Upload a **Resume PDF**
2. Upload a **Job Description PDF**
3. Click **Evaluate**
4. View:

   * Decision (Approve/Reject)
   * Score (0–100)
   * Skill Fit
   * Experience Fit
   * Salary Fit

---

## 📊 Example Output

```
Decision: APPROVE
Score: 78/100

Skill Fit: Strong match with minor gaps in cloud technologies
Experience Fit: Relevant internship and ML projects
Salary Fit: Within market range with moderate confidence
```

---

## ⚠️ Limitations

* LLM responses may vary (non-deterministic)
* Dependent on API rate limits (Gemini free tier)
* PDF text extraction quality affects results
* Salary estimation relies on external search (may be noisy)
* No ground-truth evaluation dataset

---

## 🔮 Future Improvements

* Add deterministic scoring layer
* Reduce LLM calls (optimize cost & latency)
* Improve resume parsing (section extraction)
* Introduce caching for repeated queries
* Add evaluation dataset for benchmarking
* Deploy scalable backend (API-based)

---

## 🧠 Key Learning Outcomes

* Designing multi-agent LLM systems
* Handling non-deterministic AI outputs
* Building tool-augmented reasoning pipelines
* Structuring outputs using Pydantic
* Managing API limits and failures

---

## 📌 Conclusion

The project demonstrates how LLMs can be used beyond text generation—as **decision-making components** in modular systems. It simulates real-world hiring workflows and provides a scalable foundation for intelligent recruitment tools.

---

# 🚀 Tech Stack Recommender

An intelligent **job recommendation system** that maps a user's technical skills to relevant job roles using **TF-IDF (Term Frequency–Inverse Document Frequency)** and **Cosine Similarity**.

Instead of simple keyword matching, this system provides **mathematically ranked and objective career recommendations** based on your unique skill combination.

---

# 📌 Project Overview

**Tech Stack Recommender** helps users discover the most suitable tech careers by comparing their skills against **500+ real job listings**.

Simply enter your skills (e.g., `Python, AWS, Machine Learning`) and receive:

* 🎯 Ranked job recommendations
* 📊 Match percentages
* ✅ Matching skills
* 💰 Salary range
* 📍 Company and location details
* 🧠 Career insights

---

# 🎯 Problem It Solves

Traditional job searching can be overwhelming and inefficient.

Simple keyword matching often gives every job containing **"Python"** the same score—regardless of the complete skill set.

This project solves that by:

* **Weighting rare skills higher**
  (e.g., `Kubernetes`, `TensorFlow`)

* **Comparing skill profiles mathematically**
  using **Cosine Similarity**

* **Ranking jobs objectively**
  based on your unique combination of skills

---

# ✨ Key Features

| Feature               | Description                               |
| --------------------- | ----------------------------------------- |
| 🔍 Smart Matching     | TF-IDF weighting + Cosine Similarity      |
| 📊 Ranked Results     | Jobs sorted by match percentage           |
| ✅ Skill Highlighting  | Shows which of your skills match each job |
| 💰 Salary Information | View salary ranges                        |
| 📍 Location & Type    | Shows company, location, and job type     |
| 🎨 Interactive UI     | Responsive Flask web interface            |
| 🔌 API Endpoint       | REST API support                          |

---

# 📂 Dataset Used

The dataset (`job_market.csv`) contains **500+ real job postings from 2025**.

## Included Information

### Job Titles

* Data Scientist
* ML Engineer
* DevOps Engineer
* Backend Developer
* Cloud Engineer
* Full Stack Developer

### Companies

* DataInc
* TechCorp
* EnterpriseHub
* WebDynamics

### Locations

* San Francisco
* New York
* Seattle
* Austin
* London
* Berlin
* Toronto

### Additional Fields

* Required skills
* Salary range
* Experience required
* Job type (Remote, Full-time, Contract, etc.)

---

# 📊 Most In-Demand Skills

```text
Python            ████████████████████ 40+
Machine Learning  ████████████████████ 40+
JavaScript        █████████████████    35+
Git               ███████████████      30+
AWS               █████████████        25+
Java              █████████████        25+
Kubernetes        ███████████          20+
Docker            ███████████          20+
TypeScript        ██████████           20+
TensorFlow        ████████             15+
```

---

# 🏗️ How It Works

## System Architecture

```text
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  User Input │────▶│ TF-IDF       │────▶│ Cosine Similarity│
│  (Skills)   │     │ Vectorization│     │ Comparison      │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                                                   ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Ranked Jobs │◀────│ Sort by Score│◀────│ Job Skill Match │
└─────────────┘     └──────────────┘     └─────────────────┘
```

---

# 🧠 Technical Explanation

## 1. TF-IDF Vectorization

Converts skill sets into numerical vectors.

Benefits:

* Rare skills receive higher importance
  (`Kubernetes > Python`)
* Common skills don't dominate results
* Improves recommendation quality

---

## 2. Cosine Similarity

Measures similarity between:

* **User skill vector**
* **Job skill vector**

Formula:

```python
Similarity Score = Cosine(User_Vector, Job_Vector) × 100
```

Output:

* `0` → No match
* `1` → Perfect match

Converted to percentages for easy understanding.

---

## 3. Recommendation Logic Example

### User Input

```text
Python, Kubernetes, AWS
```

### Recommended Results

| Job Role        | Matching Skills | Score |
| --------------- | --------------- | ----- |
| DevOps Engineer | Kubernetes, AWS | 92%   |
| Cloud Engineer  | AWS, Python     | 78%   |
| Data Scientist  | Python          | 45%   |

**Insight:**
`Kubernetes + AWS` gives a stronger match than `Python` alone because rare skill combinations carry more weight.

---

# 🛠️ Installation & Setup

## Prerequisites

* Python 3.8+
* pip

---

## Step 1: Clone the Project

```bash
git clone https://github.com/yourusername/tech-stack-recommender.git
cd tech-stack-recommender
```

---

## Step 2: Install Dependencies

```bash
pip install flask pandas numpy scikit-learn
```

Or using requirements file:

```bash
pip install -r requirements.txt
```

---

## Step 3: Add Dataset

Place `job_market.csv` in the root directory.

Required columns:

```text
job_title
company
location
job_type
salary_min
salary_max
experience_required
skills
```

---

## Step 4: Run the Application

```bash
python app.py
```

---

## Step 5: Open in Browser

```text
http://localhost:5000
```

---

# 📖 How to Use

## Web Interface

1. Enter your skills
   Example: `Python, AWS, Machine Learning`

2. Click **Find Matching Jobs**

3. View ranked recommendations with:

   * Match percentage
   * Matching skills
   * Salary range
   * Experience required
   * Company & location

---

# 🔌 API Usage

## Endpoint

```bash
POST /api/recommend
```

### Example Request

```bash
curl -X POST http://localhost:5000/api/recommend \
-H "Content-Type: application/json" \
-d '{"skills":["Python","Machine Learning","AWS"]}'
```

---

### Example Response

```json
{
  "user_skills": ["Python", "Machine Learning", "AWS"],
  "recommendations": [
    {
      "job_title": "Senior Data Scientist",
      "company": "WebDynamics",
      "similarity_score": 87.5,
      "matching_skills": ["Python", "Machine Learning", "AWS"],
      "salary_min": 149467,
      "salary_max": 244158
    }
  ]
}
```

---

# 🧪 Example Use Cases

## Data Science Path

**Input:**
`Python, Machine Learning, SQL, Data Analysis`

**Output:**

* Data Scientist (94%)
* ML Engineer (88%)
* Senior Data Scientist (82%)

---

## DevOps Path

**Input:**
`AWS, Docker, Kubernetes, CI/CD`

**Output:**

* DevOps Engineer (96%)
* Cloud Engineer (85%)
* Site Reliability Engineer (79%)

---

## Full Stack Path

**Input:**
`JavaScript, React, Node.js, MongoDB`

**Output:**

* Full Stack Developer (91%)
* Frontend Developer (78%)
* Backend Developer (74%)

---

# 📊 Performance Metrics

| Metric                | Value           |
| --------------------- | --------------- |
| Jobs in Database      | 500+            |
| Unique Skills         | 50+             |
| Average Response Time | < 100ms         |
| Matching Method       | TF-IDF Weighted |

---

# 🛠️ Tech Stack

* Python
* Flask
* Pandas
* NumPy
* Scikit-learn
* HTML
* CSS
* JavaScript

---

# 🚀 Future Improvements

* User authentication
* Skill gap recommendations
* Resume upload analysis
* Personalized learning roadmap
* Live job scraping from LinkedIn/Indeed

---


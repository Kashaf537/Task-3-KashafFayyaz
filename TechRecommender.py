from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

app = Flask(__name__)

class TechStackRecommender:
    def __init__(self, csv_path):
        """Initialize the recommender with job data"""
        self.df = pd.read_csv("D:\OneDrive\Desktop\OOP Project\job_market.csv")
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenize_skills, lowercase=False)
        self.skill_matrix = None
        self.job_skill_strings = None
        self.prepare_data()
    
    def tokenize_skills(self, skills_string):
        """Tokenize skills from comma-separated string"""
        if pd.isna(skills_string):
            return []
        # Split by comma and strip whitespace
        skills = [s.strip() for s in str(skills_string).split(',')]
        return skills
    
    def prepare_data(self):
        """Prepare the skill matrix using TF-IDF"""
        # Filter only Technology jobs (or all jobs with skills)
        self.df = self.df.dropna(subset=['skills'])
        self.df = self.df[self.df['skills'].str.strip() != '']
        
        # Create skill strings for vectorization
        self.job_skill_strings = self.df['skills'].fillna('').tolist()
        
        # Create TF-IDF matrix
        self.skill_matrix = self.vectorizer.fit_transform(self.job_skill_strings)
        
        print(f"Prepared {len(self.df)} jobs with {len(self.vectorizer.get_feature_names_out())} unique skills")
    
    def get_skill_frequency(self):
        """Get frequency of each skill across all jobs"""
        skills = self.vectorizer.get_feature_names_out()
        frequencies = self.skill_matrix.sum(axis=0).A1
        skill_freq = dict(zip(skills, frequencies))
        return dict(sorted(skill_freq.items(), key=lambda x: x[1], reverse=True)[:20])
    
    def recommend(self, user_skills, top_n=10):
        """Recommend jobs based on user skills"""
        # Prepare user skills string
        user_skills_str = ', '.join(user_skills)
        
        # Transform user skills using the same vectorizer
        user_vector = self.vectorizer.transform([user_skills_str])
        
        # Calculate cosine similarity with all jobs
        similarities = cosine_similarity(user_vector, self.skill_matrix).flatten()
        
        # Get top N indices
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        # Prepare results
        recommendations = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include if there's some match
                job = self.df.iloc[idx]
                recommendations.append({
                    'job_title': job['job_title'],
                    'company': job['company'],
                    'location': job['location'],
                    'job_type': job.get('job_type', 'Not specified'),
                    'salary_min': int(job['salary_min']) if pd.notna(job['salary_min']) else 'N/A',
                    'salary_max': int(job['salary_max']) if pd.notna(job['salary_max']) else 'N/A',
                    'experience_required': int(job['experience_required']) if pd.notna(job['experience_required']) else 'N/A',
                    'skills': job['skills'],
                    'similarity_score': round(similarities[idx] * 100, 2),
                    'matching_skills': self.get_matching_skills(user_skills, job['skills'])
                })
        
        return recommendations
    
    def get_matching_skills(self, user_skills, job_skills):
        """Find which user skills match the job requirements"""
        job_skill_set = set([s.strip() for s in str(job_skills).split(',')])
        return [skill for skill in user_skills if skill in job_skill_set]
    
    def get_unique_skills(self, limit=50):
        """Get all unique skills from the dataset"""
        all_skills = set()
        for skills_str in self.df['skills']:
            if pd.notna(skills_str):
                for skill in skills_str.split(','):
                    all_skills.add(skill.strip())
        return sorted(list(all_skills))[:limit]

# HTML Templates as strings
INDEX_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tech Stack Recommender</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .main-card {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 30px;
        }
        
        .input-section {
            margin-bottom: 30px;
        }
        
        .input-section label {
            display: block;
            font-weight: 600;
            margin-bottom: 10px;
            color: #333;
            font-size: 1.1rem;
        }
        
        .skills-input {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1rem;
            transition: all 0.3s;
        }
        
        .skills-input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .skills-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 15px;
            margin-bottom: 20px;
        }
        
        .skill-tag {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9rem;
        }
        
        .remove-skill {
            cursor: pointer;
            font-weight: bold;
            font-size: 1.2rem;
        }
        
        .common-skills {
            margin-top: 20px;
            padding: 20px;
            background: #f5f5f5;
            border-radius: 10px;
        }
        
        .common-skills h3 {
            margin-bottom: 15px;
            color: #333;
        }
        
        .skill-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        
        .skill-btn {
            background: white;
            border: 2px solid #667eea;
            color: #667eea;
            padding: 6px 12px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 0.85rem;
        }
        
        .skill-btn:hover {
            background: #667eea;
            color: white;
        }
        
        .recommend-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.1rem;
            border-radius: 10px;
            cursor: pointer;
            transition: transform 0.2s;
            width: 100%;
            font-weight: 600;
        }
        
        .recommend-btn:hover {
            transform: translateY(-2px);
        }
        
        .error {
            background: #fee;
            color: #c33;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .skill-frequency {
            margin-top: 30px;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 10px;
        }
        
        .skill-frequency h3 {
            margin-bottom: 15px;
        }
        
        .freq-item {
            display: inline-block;
            margin: 5px;
            padding: 5px 10px;
            background: white;
            border-radius: 15px;
            font-size: 0.85rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .freq-count {
            background: #667eea;
            color: white;
            border-radius: 10px;
            padding: 2px 6px;
            margin-left: 5px;
            font-size: 0.7rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Tech Stack Recommender</h1>
            <p>Enter your skills and get personalized job recommendations</p>
        </div>
        
        <div class="main-card">
            <form method="POST" action="/recommend" id="recommendForm">
                <div class="input-section">
                    <label>Your Skills (comma-separated)</label>
                    <input type="text" 
                           id="skillInput" 
                           class="skills-input" 
                           placeholder="e.g., Python, Machine Learning, AWS, Docker"
                           onkeypress="handleEnter(event)">
                    <div class="skills-container" id="skillsContainer"></div>
                    
                    {% for skill in skill_list %}
                    <input type="hidden" name="skills" value="{{ skill }}">
                    {% endfor %}
                </div>
                
                <div class="common-skills">
                    <h3>🔥 Popular Skills</h3>
                    <div class="skill-buttons">
                        {% for skill in common_skills %}
                        <button type="button" class="skill-btn" onclick="addSkill('{{ skill }}')">
                            {{ skill }}
                        </button>
                        {% endfor %}
                    </div>
                </div>
                
                {% if error %}
                <div class="error">{{ error }}</div>
                {% endif %}
                
                <button type="submit" class="recommend-btn">🔍 Find Matching Jobs</button>
            </form>
            
            <div class="skill-frequency">
                <h3>📊 Most In-Demand Skills (in this dataset)</h3>
                {% for skill, count in skill_freq.items() %}
                <span class="freq-item">
                    {{ skill }}
                    <span class="freq-count">{{ count }}</span>
                </span>
                {% endfor %}
            </div>
        </div>
    </div>
    
    <script>
        let skills = [];
        
        function addSkill(skill) {
            if (!skills.includes(skill)) {
                skills.push(skill);
                updateSkillsDisplay();
            }
            document.getElementById('skillInput').value = '';
        }
        
        function removeSkill(skill) {
            skills = skills.filter(s => s !== skill);
            updateSkillsDisplay();
        }
        
        function updateSkillsDisplay() {
            const container = document.getElementById('skillsContainer');
            container.innerHTML = '';
            
            skills.forEach(skill => {
                const tag = document.createElement('div');
                tag.className = 'skill-tag';
                tag.innerHTML = `
                    ${skill}
                    <span class="remove-skill" onclick="removeSkill('${skill}')">×</span>
                `;
                container.appendChild(tag);
            });
            
            // Update hidden inputs
            const form = document.getElementById('recommendForm');
            const existingInputs = form.querySelectorAll('input[name="skills"]');
            existingInputs.forEach(input => input.remove());
            
            skills.forEach(skill => {
                const input = document.createElement('input');
                input.type = 'hidden';
                input.name = 'skills';
                input.value = skill;
                form.appendChild(input);
            });
        }
        
        function handleEnter(event) {
            if (event.key === 'Enter') {
                event.preventDefault();
                const input = event.target;
                const skill = input.value.trim();
                if (skill) {
                    addSkill(skill);
                }
            }
        }
        
        {% if skill_list %}
        skills = {{ skill_list|tojson }};
        updateSkillsDisplay();
        {% endif %}
    </script>
</body>
</html>
'''

RESULTS_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recommendations - Tech Stack Recommender</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .user-skills {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
        }
        
        .skill-tag {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 20px;
            margin: 5px;
            font-weight: 500;
        }
        
        .back-btn {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            text-decoration: none;
            margin-top: 20px;
            transition: transform 0.2s;
        }
        
        .back-btn:hover {
            transform: translateX(-5px);
        }
        
        .recommendations {
            display: grid;
            gap: 20px;
        }
        
        .job-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            transition: transform 0.2s, box-shadow 0.2s;
            cursor: pointer;
        }
        
        .job-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .job-header {
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        
        .job-title {
            font-size: 1.4rem;
            color: #333;
            font-weight: 600;
        }
        
        .similarity-score {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        
        .company {
            color: #667eea;
            font-weight: 500;
            margin-bottom: 10px;
        }
        
        .job-details {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 15px;
            color: #666;
            font-size: 0.9rem;
        }
        
        .detail-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .skills-section {
            margin-top: 15px;
        }
        
        .skills-label {
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
        }
        
        .skill-badge {
            display: inline-block;
            background: #f0f0f0;
            padding: 5px 12px;
            border-radius: 15px;
            margin: 3px;
            font-size: 0.85rem;
        }
        
        .skill-badge.matching {
            background: #4caf50;
            color: white;
        }
        
        .matching-skills {
            margin-top: 10px;
            padding: 10px;
            background: #e8f5e9;
            border-radius: 10px;
        }
        
        .no-results {
            text-align: center;
            padding: 50px;
            background: white;
            border-radius: 20px;
        }
        
        @media (max-width: 768px) {
            .job-header {
                flex-direction: column;
                gap: 10px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <a href="/" class="back-btn">← Back to Search</a>
            
            <div class="user-skills">
                <h3>🎯 Your Skills</h3>
                {% for skill in user_skills %}
                <span class="skill-tag">{{ skill }}</span>
                {% endfor %}
            </div>
        </div>
        
        <div class="recommendations">
            <h2 style="color: white; margin-bottom: 10px;">
                📋 Top Job Recommendations
                <span style="font-size: 0.9rem;">({{ recommendations|length }} matches)</span>
            </h2>
            
            {% if recommendations %}
                {% for job in recommendations %}
                <div class="job-card">
                    <div class="job-header">
                        <div>
                            <div class="job-title">{{ job.job_title }}</div>
                            <div class="company">{{ job.company }}</div>
                        </div>
                        <div class="similarity-score">
                            Match: {{ job.similarity_score }}%
                        </div>
                    </div>
                    
                    <div class="job-details">
                        <span class="detail-item">📍 {{ job.location }}</span>
                        <span class="detail-item">💼 {{ job.job_type }}</span>
                        <span class="detail-item">
                            💰 ${{ job.salary_min }} - ${{ job.salary_max }}
                        </span>
                        <span class="detail-item">
                            📅 {{ job.experience_required }} years exp
                        </span>
                    </div>
                    
                    <div class="skills-section">
                        <div class="skills-label">Required Skills:</div>
                        {% set job_skills = job.skills.split(',') %}
                        {% for skill in job_skills %}
                            {% set skill_clean = skill.strip() %}
                            {% if skill_clean in user_skills %}
                            <span class="skill-badge matching">{{ skill_clean }} ✓</span>
                            {% else %}
                            <span class="skill-badge">{{ skill_clean }}</span>
                            {% endif %}
                        {% endfor %}
                    </div>
                    
                    {% if job.matching_skills %}
                    <div class="matching-skills">
                        ✅ Your matching skills: {{ job.matching_skills|join(', ') }}
                    </div>
                    {% endif %}
                </div>
                {% endfor %}
            {% else %}
                <div class="no-results">
                    <h3>😅 No matching jobs found</h3>
                    <p>Try adding more skills or different technologies</p>
                    <a href="/" style="display: inline-block; margin-top: 20px; padding: 10px 20px; background: #667eea; color: white; text-decoration: none; border-radius: 10px;">
                        Try Again
                    </a>
                </div>
            {% endif %}
        </div>
        
        <div style="margin-top: 30px; text-align: center; color: white; opacity: 0.8; font-size: 0.85rem;">
            <p>Based on TF-IDF matching | Higher match % = better skill alignment</p>
        </div>
    </div>
</body>
</html>
'''

# Initialize recommender
def create_app():
    # Check if CSV file exists
    if not os.path.exists('job_market.csv'):
        print("ERROR: job_market.csv not found in the current directory!")
        print("Please make sure the CSV file is in the same folder as this script.")
        return None
    
    return TechStackRecommender('job_market.csv')

# Flask routes
@app.route('/')
def index():
    if not hasattr(app, 'recommender'):
        return "Error: Could not load recommender. Make sure job_market.csv exists."
    
    skill_freq = app.recommender.get_skill_frequency()
    common_skills = list(skill_freq.keys())[:15]
    skill_list = []
    return render_template_string(INDEX_HTML, 
                                common_skills=common_skills, 
                                skill_freq=skill_freq,
                                skill_list=skill_list,
                                error=None)

@app.route('/recommend', methods=['POST'])
def recommend():
    if not hasattr(app, 'recommender'):
        return "Error: Could not load recommender. Make sure job_market.csv exists."
    
    user_skills = request.form.getlist('skills')
    user_skills = [s.strip() for s in user_skills if s.strip()]
    
    if not user_skills:
        skill_freq = app.recommender.get_skill_frequency()
        common_skills = list(skill_freq.keys())[:15]
        return render_template_string(INDEX_HTML, 
                                    common_skills=common_skills,
                                    skill_freq=skill_freq,
                                    skill_list=[],
                                    error="Please enter at least one skill")
    
    # Get recommendations
    recommendations = app.recommender.recommend(user_skills, top_n=15)
    
    return render_template_string(RESULTS_HTML, 
                                recommendations=recommendations, 
                                user_skills=user_skills)

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """JSON API endpoint for recommendations"""
    if not hasattr(app, 'recommender'):
        return jsonify({'error': 'Recommender not loaded'}), 500
    
    data = request.get_json()
    user_skills = data.get('skills', [])
    
    recommendations = app.recommender.recommend(user_skills, top_n=10)
    
    return jsonify({
        'user_skills': user_skills,
        'recommendations': recommendations
    })

@app.route('/skills')
def get_skills():
    """Get all available skills"""
    if not hasattr(app, 'recommender'):
        return jsonify({'error': 'Recommender not loaded'}), 500
    
    all_skills = app.recommender.get_unique_skills()
    return jsonify({'skills': all_skills})

if __name__ == '__main__':
    # Initialize the recommender
    recommender = create_app()
    
    if recommender:
        app.recommender = recommender
        print("\n" + "="*50)
        print("🚀 Tech Stack Recommender is running!")
        print("="*50)
        print("📍 Open your browser and go to: http://localhost:5000")
        print("📁 Make sure job_market.csv is in the same directory")
        print("="*50 + "\n")
        app.run(debug=True, port=5000)
    else:
        print("\n❌ Failed to start the application. Please check:")
        print("1. You have 'job_market.csv' in the same folder")
        print("2. The CSV file has the required columns")
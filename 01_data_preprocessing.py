# ================================================================
# KAZAKHSTAN TECH LABOR MARKET ANALYSIS
# Final Project - Data Mining Course
# ================================================================
# Author: Azamat
# Course: Data Mining, KBTU
# Dataset: hh.kz (HeadHunter Kazakhstan API)
# Problems: 4 research questions on skill valuation, salaries, 
#           regional differences, and market-education alignment
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from datetime import datetime
from collections import Counter

# ML Libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score
)
from scipy import stats
from scipy.stats import f_oneway, ttest_ind

# Association Rules
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
except ImportError:
    print("⚠️  mlxtend not installed. Run: pip install mlxtend")

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("  KAZAKHSTAN TECH LABOR MARKET ANALYSIS")
print("  Data Mining Final Project - KBTU")
print("="*70)

# PART 1: DATA LOADING & INITIAL EXPLORATION

class DataLoader:
    """Load and perform initial exploration of job market data"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        
    def load_data(self):
        """Load CSV data and show basic info"""
        print("\n📂 LOADING DATA...")
        print("-" * 70)
        
        self.df = pd.read_csv(self.filepath)
        
        print(f"✅ Data loaded successfully!")
        print(f"   📊 Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"   📅 Date range: {self.df['published_at'].min()} to {self.df['published_at'].max()}")
        
        return self.df
    
    def show_info(self):
        """Display dataset information"""
        print("\n📋 DATASET INFO:")
        print("-" * 70)
        print(self.df.info())
        
        print("\n📊 BASIC STATISTICS:")
        print("-" * 70)
        print(self.df.describe())
        
        print("\n❓ MISSING VALUES:")
        print("-" * 70)
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct
        }).sort_values('Percentage', ascending=False)
        print(missing_df[missing_df['Missing Count'] > 0])
        
        return missing_df


# PART 2: DATA PREPROCESSING & FEATURE ENGINEERING

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline
    Covers Lecture #2: Data Preprocessing, Aggregation, Feature Creation
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_size = len(df)
        
    def clean_salaries(self):
        """
        Normalize salaries to KZT/month
        Handle currency conversion and salary ranges
        """
        print("\n💰 SALARY NORMALIZATION...")
        print("-" * 70)
        
        def normalize_salary(row):
            """Convert all salaries to KZT monthly"""
            salary_from = row['salary_from']
            salary_to = row['salary_to']
            currency = str(row['salary_currency']).upper()
            
            # If both missing, return NaN
            if pd.isna(salary_from) and pd.isna(salary_to):
                return np.nan
            
            # Calculate average or use available value
            if pd.notna(salary_from) and pd.notna(salary_to):
                salary = (salary_from + salary_to) / 2
            elif pd.notna(salary_from):
                salary = salary_from
            else:
                salary = salary_to
            
            # Currency conversion to KZT
            if 'USD' in currency or currency == 'USD':
                salary *= 480  # 1 USD ≈ 480 KZT (approximate)
            elif 'EUR' in currency or currency == 'EUR':
                salary *= 520  # 1 EUR ≈ 520 KZT
            elif 'RUB' in currency or 'RUR' in currency or currency == 'RUB':
                salary *= 5.2  # 1 RUB ≈ 5.2 KZT
            # If already KZT, no conversion needed
            
            return salary
        
        # Apply normalization
        self.df['salary_monthly_kzt'] = self.df.apply(normalize_salary, axis=1)
        
        # Statistics
        with_salary = self.df['salary_monthly_kzt'].notna().sum()
        missing_salary = self.df['salary_monthly_kzt'].isna().sum()
        missing_pct = (missing_salary / len(self.df)) * 100
        
        print(f"   ✅ Salaries normalized to KZT/month")
        print(f"   📊 Jobs with salary: {with_salary} ({100-missing_pct:.1f}%)")
        print(f"   ❌ Jobs without salary: {missing_salary} ({missing_pct:.1f}%)")
        print(f"   💵 Salary range: {self.df['salary_monthly_kzt'].min():,.0f} - {self.df['salary_monthly_kzt'].max():,.0f} KZT")
        print(f"   📈 Mean salary: {self.df['salary_monthly_kzt'].mean():,.0f} KZT")
        print(f"   📊 Median salary: {self.df['salary_monthly_kzt'].median():,.0f} KZT")
        
        return self
    
    def normalize_locations(self):
        """
        Standardize location names
        Map variations to standard city names
        """
        print("\n📍 LOCATION NORMALIZATION...")
        print("-" * 70)
        
        def normalize_area(area_name):
            """Standardize city names"""
            if pd.isna(area_name):
                return 'Other'
            
            area = str(area_name).lower()
            
            # Almaty variations
            if any(x in area for x in ['алматы', 'almaty', 'алма-ата']):
                return 'Almaty'
            
            # Astana variations (also known as Nur-Sultan)
            if any(x in area for x in ['астана', 'astana', 'нур-султан', 'nur-sultan']):
                return 'Astana'
            
            # Shymkent
            if any(x in area for x in ['шымкент', 'shymkent', 'чимкент']):
                return 'Shymkent'
            
            # Other major cities
            if any(x in area for x in ['караганд', 'karagand']):
                return 'Karaganda'
            if any(x in area for x in ['актобе', 'aktobe', 'aqtobe']):
                return 'Aktobe'
            if any(x in area for x in ['тараз', 'taraz']):
                return 'Taraz'
            
            return 'Other'
        
        self.df['area_normalized'] = self.df['area'].apply(normalize_area)
        
        # Statistics
        print("   ✅ Locations normalized")
        print("\n   📊 Distribution by city:")
        print(self.df['area_normalized'].value_counts().to_string())
        
        return self
    
    def encode_experience(self):
        """
        Convert experience levels to ordinal encoding
        Map Russian text to numeric values
        """
        print("\n👔 EXPERIENCE ENCODING...")
        print("-" * 70)
        
        # Experience mapping (ordinal)
        exp_mapping = {
            'Нет опыта': 0,
            'От 1 года до 3 лет': 1,
            'От 3 до 6 лет': 2,
            'Более 6 лет': 3
        }
        
        self.df['experience_level'] = self.df['experience'].map(exp_mapping)
        
        # Fill missing with 0 (assume entry-level)
        self.df['experience_level'].fillna(0, inplace=True)
        
        print("   ✅ Experience encoded to ordinal values")
        print("\n   📊 Distribution:")
        exp_dist = self.df.groupby('experience').agg({
            'experience_level': 'first',
            'vacancy_id': 'count'
        }).rename(columns={'vacancy_id': 'count'})
        print(exp_dist.to_string())
        
        return self
    
    def extract_skills_nlp(self):
        """
        Extract skills from job descriptions using NLP
        Covers Lecture #12: Text Mining and NLP
        """
        print("\n🔍 SKILL EXTRACTION (NLP)...")
        print("-" * 70)
        
        # Comprehensive skill dictionary (200+ skills)
        TECH_SKILLS = [
            # Programming Languages
            'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'Go', 'Rust',
            'PHP', 'Ruby', 'Swift', 'Kotlin', 'Scala', 'R', 'MATLAB',
            
            # Web Frontend
            'React', 'Vue', 'Vue.js', 'Angular', 'HTML', 'CSS', 'SASS', 'LESS',
            'jQuery', 'Bootstrap', 'Tailwind', 'Next.js', 'Nuxt.js',
            
            # Web Backend
            'Django', 'Flask', 'FastAPI', 'Node.js', 'Express', 'Spring', 'Spring Boot',
            'Laravel', 'ASP.NET', 'Ruby on Rails', 'Nest.js',
            
            # Databases
            'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Elasticsearch',
            'Oracle', 'SQL Server', 'SQLite', 'Cassandra', 'DynamoDB',
            
            # DevOps & Cloud
            'Docker', 'Kubernetes', 'AWS', 'Azure', 'Google Cloud', 'GCP',
            'Jenkins', 'GitLab CI', 'GitHub Actions', 'Terraform', 'Ansible',
            'Linux', 'Nginx', 'Apache',
            
            # Mobile
            'Android', 'iOS', 'React Native', 'Flutter', 'Xamarin',
            
            # Data Science & ML
            'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 
            'Scikit-learn', 'Pandas', 'NumPy', 'Jupyter', 'Data Science',
            'Computer Vision', 'NLP', 'Natural Language Processing',
            
            # Tools & Methodologies
            'Git', 'GitHub', 'GitLab', 'Jira', 'Confluence', 'Agile', 'Scrum',
            'REST API', 'GraphQL', 'Microservices', 'CI/CD', 'TDD', 'Unit Testing',
            
            # Other
            'Blockchain', 'Solidity', 'Ethereum', 'Smart Contracts',
            'Unity', 'Unreal Engine', '3D Graphics',
            'Photoshop', 'Figma', 'UI/UX', 'Design'
        ]
        
        def extract_skills_from_text(text):
            """Extract skills from job description"""
            if pd.isna(text):
                return []
            
            text = str(text).lower()
            found_skills = []
            
            for skill in TECH_SKILLS:
                # Case-insensitive search with word boundaries
                pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(pattern, text):
                    found_skills.append(skill)
            
            return found_skills
        
        # Extract from description
        print("   🔄 Extracting skills from descriptions...")
        self.df['skills_extracted'] = self.df['description'].apply(extract_skills_from_text)
        
        # Also parse key_skills column
        def parse_key_skills(x):
            """Parse key_skills column"""
            if pd.isna(x):
                return []
            
            # If it's a list representation as string
            if '[' in str(x):
                skills = re.findall(r"'([^']*)'", str(x))
                return skills
            
            # If comma-separated
            return [s.strip() for s in str(x).split(',') if s.strip()]
        
        self.df['skills_key_parsed'] = self.df['key_skills'].apply(parse_key_skills)
        
        # Combine both sources
        self.df['skills_all'] = self.df.apply(
            lambda row: list(set(row['skills_extracted'] + row['skills_key_parsed'])),
            axis=1
        )
        
        # Statistics
        total_skills = sum(len(skills) for skills in self.df['skills_all'])
        avg_skills = total_skills / len(self.df)
        jobs_with_skills = self.df['skills_all'].apply(len).gt(0).sum()
        
        print(f"   ✅ Skill extraction complete")
        print(f"   📊 Total skills extracted: {total_skills}")
        print(f"   📈 Average skills per job: {avg_skills:.1f}")
        print(f"   💼 Jobs with at least 1 skill: {jobs_with_skills} ({jobs_with_skills/len(self.df)*100:.1f}%)")
        
        # Show top skills
        all_skills_flat = [skill for skills in self.df['skills_all'] for skill in skills]
        skill_counts = Counter(all_skills_flat)
        
        print("\n   🏆 TOP 20 MOST DEMANDED SKILLS:")
        for skill, count in skill_counts.most_common(20):
            pct = (count / len(self.df)) * 100
            print(f"      {skill:25s} : {count:4d} jobs ({pct:5.1f}%)")
        
        return self
    
    def create_skill_features(self, top_n=50):
        """
        Create binary features for top N skills
        Feature Engineering for ML models
        """
        print(f"\n🎯 CREATING SKILL FEATURES (TOP {top_n})...")
        print("-" * 70)
        
        # Get top N skills
        all_skills_flat = [skill for skills in self.df['skills_all'] for skill in skills]
        skill_counts = Counter(all_skills_flat)
        top_skills = [skill for skill, count in skill_counts.most_common(top_n)]
        
        # Create binary columns
        for skill in top_skills:
            col_name = f'skill_{skill.replace(" ", "_").replace(".", "").lower()}'
            self.df[col_name] = self.df['skills_all'].apply(lambda x: 1 if skill in x else 0)
        
        self.skill_columns = [f'skill_{skill.replace(" ", "_").replace(".", "").lower()}' for skill in top_skills]
        
        print(f"   ✅ Created {len(self.skill_columns)} binary skill features")
        print(f"   📊 Skill features cover {len(top_skills)} most common skills")
        
        return self
    
    def handle_outliers(self, method='iqr'):
        """
        Detect and handle salary outliers
        Covers Lecture #4: Outlier Analysis
        """
        print("\n📉 OUTLIER DETECTION & HANDLING...")
        print("-" * 70)
        
        salaries = self.df['salary_monthly_kzt'].dropna()
        
        if method == 'iqr':
            Q1 = salaries.quantile(0.25)
            Q3 = salaries.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[
                (self.df['salary_monthly_kzt'] < lower_bound) | 
                (self.df['salary_monthly_kzt'] > upper_bound)
            ]
            
            print(f"   📊 IQR Method:")
            print(f"      Q1 (25%): {Q1:,.0f} KZT")
            print(f"      Q3 (75%): {Q3:,.0f} KZT")
            print(f"      IQR: {IQR:,.0f} KZT")
            print(f"      Lower Bound: {lower_bound:,.0f} KZT")
            print(f"      Upper Bound: {upper_bound:,.0f} KZT")
            print(f"   🔴 Outliers detected: {len(outliers)} ({len(outliers)/len(salaries)*100:.1f}%)")
            
            # Cap outliers instead of removing (preserve data)
            self.df.loc[self.df['salary_monthly_kzt'] < lower_bound, 'salary_monthly_kzt'] = lower_bound
            self.df.loc[self.df['salary_monthly_kzt'] > upper_bound, 'salary_monthly_kzt'] = upper_bound
            
            print(f"   ✅ Outliers capped to bounds")
        
        return self
    
    def get_processed_data(self):
        """Return processed dataframe"""
        print(f"\n✅ PREPROCESSING COMPLETE!")
        print(f"   Original size: {self.original_size} rows")
        print(f"   Final size: {len(self.df)} rows")
        print(f"   Features created: {len(self.df.columns)} columns")
        
        return self.df



if __name__ == "__main__":
    # Load data
    loader = DataLoader('fixed_vacancies.csv')
    df_raw = loader.load_data()
    loader.show_info()
    
    # Preprocess data
    preprocessor = DataPreprocessor(df_raw)
    df_processed = (preprocessor
                    .clean_salaries()
                    .normalize_locations()
                    .encode_experience()
                    .extract_skills_nlp()
                    .create_skill_features(top_n=50)
                    .handle_outliers(method='iqr')
                    .get_processed_data())
    
    # Save processed data
    output_file = 'data_processed.csv'
    df_processed.to_csv(output_file, index=False)
    print(f"\n💾 Processed data saved to: {output_file}")
    
    print("\n" + "="*70)
    print("  PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("  Ready for EDA and ML modeling")
    print("="*70)

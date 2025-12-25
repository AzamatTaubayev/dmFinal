import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

try:
    from colorama import Fore, Back, Style, init

    # Initialize colorama for colored terminal output
    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    # Fallback if colorama not installed
    HAS_COLOR = False


    class Fore:
        CYAN = YELLOW = GREEN = RED = WHITE = ""


    class Style:
        RESET_ALL = ""

print(Fore.CYAN + "=" * 80)
print(Fore.CYAN + "  🎓 KAZAKHSTAN TECH CAREER ADVISOR FOR KBTU STUDENTS 🚀")
print(Fore.CYAN + "=" * 80)

# LOAD DATA AND TRAIN MODEL


print("\n📂 Loading market data and training model...")

try:
    # Load market data
    df_market = pd.read_csv('data_with_imputed_salaries.csv')

    # Load skill premiums
    skill_premiums = pd.read_csv('skill_premiums.csv')

    print(Fore.GREEN + "✓ Market data loaded: {} jobs".format(len(df_market)))
    print(Fore.GREEN + "✓ Skill premiums loaded: {} skills".format(len(skill_premiums)))

    # Train simple model on the fly
    # Train simple model on the fly
    print(Fore.YELLOW + "\n🔄 Training salary prediction model...")

    # Get skill columns
    skill_cols = [col for col in df_market.columns if col.startswith('skill_')]

    # Prepare features (only jobs with known salaries)
    df_train = df_market[df_market['salary_monthly_kzt'].notna()].copy()

    # Include experience_level if available
    feature_cols = skill_cols.copy()
    if 'experience_level' in df_train.columns:
        feature_cols.append('experience_level')
        print(Fore.GREEN + "✓ Using experience_level in model")

    X = df_train[feature_cols]
    y = df_train['salary_monthly_kzt']

    # Train Ridge model
    model = Ridge(alpha=1.0)
    model.fit(X, y)

    feature_list = feature_cols

    print(Fore.GREEN + "✓ Model trained successfully!")
    print(Fore.GREEN + "✓ Training data: {} jobs with known salaries".format(len(df_train)))

except FileNotFoundError as e:
    print(Fore.RED + f"\n❌ ERROR: Required files not found!")
    print(Fore.RED + f"   Please run '03_problem1_2_skill_valuation_salary_prediction.py' first!")
    print(Fore.RED + f"   This will generate:")
    print(Fore.RED + f"   - skill_premiums.csv")
    print(Fore.RED + f"   - data_with_imputed_salaries.csv")
    exit(1)

# DEFINE SKILL CATALOG

# All available skills (72 total)
ALL_SKILLS = [
    'python', 'java', 'javascript', 'typescript', 'go', 'c++', 'c#', 'php', 'ruby', 'swift',
    'kotlin', 'scala', 'rust', 'r', 'matlab',
    'react', 'vue', 'angular', 'html', 'css', 'sass', 'jquery', 'bootstrap',
    'django', 'spring', 'flask', 'fastapi', 'nodejs', 'express',
    'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'oracle',
    'docker', 'kubernetes', 'ci/cd', 'jenkins', 'gitlab_ci', 'github_actions',
    'aws', 'azure', 'gcp', 'terraform', 'ansible',
    'git', 'gitlab', 'github', 'bitbucket',
    'linux', 'unix', 'bash', 'powershell',
    'agile', 'scrum', 'kanban', 'jira', 'confluence',
    'machine_learning', 'deep_learning', 'tensorflow', 'pytorch', 'data_mining',
    'sql', 'nosql', 'graphql', 'rest_api', 'microservices',
    'android', 'ios', 'flutter', 'react_native'
]

# KBTU curriculum (taught skills)
KBTU_TAUGHT = [
    'python', 'java', 'javascript', 'typescript', 'go',
    'docker', 'kubernetes', 'ci/cd', 'gitlab_ci', 'jenkins', 'linux',
    'git', 'github', 'gitlab', 'agile', 'scrum', 'jira',
    'react', 'vue', 'angular', 'html', 'css',
    'android', 'ios', 'django', 'spring', 'rest_api',
    'postgresql', 'mysql', 'sql',
    'machine_learning', 'deep_learning', 'data_mining',
    'aws', 'azure'
]

# Critical gaps (not taught at KBTU but high demand)
CRITICAL_GAPS = ['redis', 'mongodb', 'elasticsearch']

# Career paths based on clustering
DEVOPS_SKILLS = ['docker', 'ci/cd', 'postgresql', 'git', 'redis', 'kubernetes', 'jenkins', 'gitlab_ci']
GENERAL_DEV_SKILLS = ['git', 'linux', 'javascript', 'python', 'agile', 'java', 'react', 'sql']


# HELPER FUNCTIONS

def get_salary_percentile(predicted_salary):
    """Calculate what percentile the predicted salary falls into"""
    percentile = (df_market['salary_monthly_kzt'] <= predicted_salary).sum() / len(df_market) * 100
    return percentile


def get_market_stats():
    """Get market salary statistics"""
    return {
        'min': df_market['salary_monthly_kzt'].min(),
        'q25': df_market['salary_monthly_kzt'].quantile(0.25),
        'median': df_market['salary_monthly_kzt'].median(),
        'q75': df_market['salary_monthly_kzt'].quantile(0.75),
        'max': df_market['salary_monthly_kzt'].max(),
        'mean': df_market['salary_monthly_kzt'].mean()
    }


def format_salary(amount):
    """Format salary with thousand separators"""
    return f"{amount:,.0f}".replace(',', ' ')


def get_career_path_recommendation(user_skills):
    """Recommend career path based on skills"""
    devops_count = sum(1 for s in user_skills if s in DEVOPS_SKILLS)
    general_count = sum(1 for s in user_skills if s in GENERAL_DEV_SKILLS)

    devops_pct = devops_count / len(DEVOPS_SKILLS) * 100
    general_pct = general_count / len(GENERAL_DEV_SKILLS) * 100

    if devops_pct > general_pct:
        return "DevOps/Infrastructure Specialist", devops_pct
    else:
        return "General Software Developer", general_pct


def get_top_premium_skills(n=10):
    """Get top N skills by premium"""
    return skill_premiums.nlargest(n, 'Coefficient')


def get_skill_gap_status(skill):
    """Check if skill is taught at KBTU or is a gap"""
    skill_lower = skill.lower().replace(' ', '_')
    if skill_lower in KBTU_TAUGHT:
        return "✅ KBTU"
    elif skill_lower in CRITICAL_GAPS:
        return "🔴 GAP"
    else:
        return "⚠️  Other"


def calculate_skill_frequency(skill_name):
    """Calculate how often a skill appears in job postings"""
    skill_col = 'skill_' + skill_name.lower().replace(' ', '_')
    if skill_col in df_market.columns:
        frequency = (df_market[skill_col] == 1).sum() / len(df_market) * 100
        return frequency
    return 0.0


# MAIN INTERACTIVE ADVISOR


def run_career_advisor():
    """Main interactive career advisor"""

    print("\n" + Fore.YELLOW + "=" * 80)
    print(Fore.YELLOW + "  WELCOME TO YOUR PERSONALIZED CAREER ADVISOR!")
    print(Fore.YELLOW + "=" * 80)

    print("\n💡 This tool will:")
    print("   1. Predict your market salary based on your current skills")
    print("   2. Show you where you stand in the market (percentile)")
    print("   3. Recommend high-value skills to learn next")
    print("   4. Suggest optimal career path (DevOps vs General Dev)")

    # STEP 1: COLLECT USER SKILLS

    print("\n" + Fore.CYAN + "=" * 80)
    print(Fore.CYAN + "STEP 1: TELL US YOUR CURRENT SKILLS")
    print(Fore.CYAN + "=" * 80)

    print("\n📚 Available skill categories:")
    print("   1. Programming Languages (Python, Java, JavaScript, TypeScript, Go, etc.)")
    print("   2. Frontend (React, Vue, Angular, HTML, CSS)")
    print("   3. Backend (Django, Spring, Flask, Node.js)")
    print("   4. Databases (PostgreSQL, MySQL, MongoDB, Redis)")
    print("   5. DevOps (Docker, Kubernetes, CI/CD, Jenkins, GitLab CI)")
    print("   6. Cloud (AWS, Azure, GCP)")
    print("   7. Tools (Git, Linux, Agile, Scrum)")
    print("   8. Mobile (Android, iOS, Flutter)")
    print("   9. Data Science (Machine Learning, Deep Learning, TensorFlow)")

    print("\n📝 Enter your skills (separated by commas):")
    print("   Example: python, javascript, react, docker, postgresql, git")
    print()

    user_input = input(Fore.GREEN + "Your skills: " + Style.RESET_ALL).lower().strip()

    # Parse user skills
    user_skills = [s.strip().replace(' ', '_') for s in user_input.split(',')]
    user_skills = [s for s in user_skills if s]  # Remove empty strings

    # Validate skills
    valid_skills = [s for s in user_skills if f'skill_{s}' in feature_list]
    invalid_skills = [s for s in user_skills if f'skill_{s}' not in feature_list]

    if invalid_skills:
        print(Fore.YELLOW + f"\n⚠️  Warning: These skills were not recognized: {', '.join(invalid_skills)}")
        print(Fore.YELLOW + f"   They will be ignored in the prediction.")

    if not valid_skills:
        print(Fore.RED + "\n❌ No valid skills entered! Please try again.")
        return

    print(Fore.GREEN + f"\n✓ Recognized {len(valid_skills)} valid skills")

    # ASK FOR EXPERIENCE LEVEL

    print("\n" + Fore.CYAN + "=" * 80)
    print(Fore.CYAN + "💼 WRITE YOUR WORK EXPERIENCE:")
    print(Fore.CYAN + "=" * 80)

    print("\n📊 Выберите уровень опыта:")
    print("   0 - No experience / Intern (0 years)")
    print("   1 - Junior (1-2 years)")
    print("   2 - Middle (3-5 years)")
    print("   3 - Senior (6+ years)")
    print()

    while True:
        experience_input = input(Fore.GREEN + "Your experience (0-3): " + Style.RESET_ALL).strip()
        if experience_input in ['0', '1', '2', '3']:
            experience_level = int(experience_input)
            break
        else:
            print(Fore.RED + "⚠️  Please, write a number between 0 and 3")

    experience_names = {
        0: "No experience / Intern (0 лет)",
        1: "Junior (1-2 года)",
        2: "Middle (3-5 лет)",
        3: "Senior (6+ лет)"
    }

    print(Fore.GREEN + f"✓ Experience: {experience_names[experience_level]}")

    print("\n📍 Choose your city:")
    print("   0 - Almaty")
    print("   1 - Astana")
    print("   2 - Other (remote/regions)")

    city_input = int(input("City (0-2): "))
    city_names = {0: 'Almaty', 1: 'Astana', 2: 'Other'}


    # STEP 2: PREDICT SALARY

    print("\n" + Fore.CYAN + "=" * 80)
    print(Fore.CYAN + "STEP 2: SALARY PREDICTION")
    print(Fore.CYAN + "=" * 80)

    # Create feature vector
    X_user = pd.DataFrame(0, index=[0], columns=feature_list)

    # Add area dummies to X_user
    area_cols = [col for col in feature_list if col.startswith('area_normalized_')]
    for col in area_cols:
        X_user[col] = 0

    if city_input == 1:  # Astana
        if 'area_normalized_Astana' in X_user.columns:
            X_user['area_normalized_Astana'] = 1
    elif city_input == 2:  # Other
        if 'area_normalized_Other' in X_user.columns:
            X_user['area_normalized_Other'] = 1

    # Add skills
    for skill in valid_skills:
        col = f'skill_{skill}'
        if col in X_user.columns:
            X_user[col] = 1

    # Add experience level
    if 'experience_level' in X_user.columns:
        X_user['experience_level'] = experience_level

    # Predict salary
    predicted_salary = model.predict(X_user)[0]

    # Get market stats
    market_stats = get_market_stats()
    percentile = get_salary_percentile(predicted_salary)

    print(f"\n💰 YOUR PREDICTED SALARY: {Fore.GREEN}{format_salary(predicted_salary)} KZT/month{Style.RESET_ALL}")
    print(f"\n📊 Market Position: {Fore.YELLOW}{percentile:.1f}th percentile{Style.RESET_ALL}")

    print("\n📈 Market Salary Distribution:")
    print(f"   Minimum:  {format_salary(market_stats['min']):>15s} KZT")
    print(f"   25th %:   {format_salary(market_stats['q25']):>15s} KZT")
    print(f"   Median:   {format_salary(market_stats['median']):>15s} KZT")
    print(f"   75th %:   {format_salary(market_stats['q75']):>15s} KZT")
    print(f"   Maximum:  {format_salary(market_stats['max']):>15s} KZT")
    print(f"   Average:  {format_salary(market_stats['mean']):>15s} KZT")
    # Show experience impact
    print(f"\n💼 Experience Impact:")
    print(f"   Current level: {experience_names[experience_level]}")

    if experience_level < 3:
        # Calculate potential with more experience
        X_next = X_user.copy()
        X_next['experience_level'] = experience_level + 1
        predicted_next = model.predict(X_next)[0]
        diff = predicted_next - predicted_salary

        print(
            f"   Potential with {experience_names[experience_level + 1]}: {Fore.YELLOW}{format_salary(predicted_next)} KZT{Style.RESET_ALL}")
        print(
            f"   Experience boost: {Fore.GREEN}+{format_salary(diff)} KZT (+{diff / predicted_salary * 100:.1f}%){Style.RESET_ALL}")
    else:
        print(f"   {Fore.GREEN}✓ You're at the highest experience level!{Style.RESET_ALL}")
    # Interpretation
    print(f"\n💡 Interpretation:")
    if percentile >= 75:
        print(Fore.GREEN + f"   🎉 EXCELLENT! You're in the top {100 - percentile:.1f}% of the market.")
        print(Fore.GREEN + f"   Your skills command a premium salary.")
    elif percentile >= 50:
        print(Fore.YELLOW + f"   👍 GOOD! You're above the market median.")
        print(Fore.YELLOW + f"   You're well-positioned, with room for growth.")
    elif percentile >= 25:
        print(Fore.YELLOW + f"   📚 DEVELOPING! You're at entry-to-mid level.")
        print(Fore.YELLOW + f"   Focus on high-value skills to increase earning potential.")
    else:
        print(Fore.RED + f"   🎯 ENTRY LEVEL! You're starting your career.")
        print(Fore.RED + f"   Learning key skills will significantly boost your salary.")

    # ================================================================
    # STEP 3: SKILL BREAKDOWN
    # ================================================================

    print("\n" + Fore.CYAN + "=" * 80)
    print(Fore.CYAN + "STEP 3: YOUR SKILL VALUE BREAKDOWN")
    print(Fore.CYAN + "=" * 80)

    # Get premiums for user's skills
    user_skill_names = [s.replace('_', ' ').title() for s in valid_skills]
    user_skill_premiums = skill_premiums[skill_premiums['Skill'].isin(user_skill_names)].copy()
    user_skill_premiums = user_skill_premiums.sort_values('Coefficient', ascending=False)

    if len(user_skill_premiums) > 0:
        print("\n💎 Value of Your Skills (Salary Premium):\n")
        print(f"{'Skill':<20} {'Premium':>15} {'KBTU Status':>15}")
        print("-" * 50)

        total_premium = 0
        for _, row in user_skill_premiums.iterrows():
            skill_name = row['Skill']
            premium = row['Coefficient']
            status = get_skill_gap_status(skill_name)

            total_premium += premium

            # Color code by premium level
            if premium >= 200000:
                color = Fore.GREEN
            elif premium >= 100000:
                color = Fore.YELLOW
            else:
                color = Fore.WHITE

            print(f"{skill_name:<20} {color}+{format_salary(premium):>14s} KZT{Style.RESET_ALL} {status:>15}")

        print("-" * 50)
        print(f"{'TOTAL PREMIUM:':<20} {Fore.CYAN}+{format_salary(total_premium):>14s} KZT{Style.RESET_ALL}")
    else:
        print(Fore.YELLOW + "\n⚠️  No premium data available for your skills.")

    # ================================================================
    # STEP 4: CAREER PATH RECOMMENDATION
    # ================================================================

    print("\n" + Fore.CYAN + "=" * 80)
    print(Fore.CYAN + "STEP 4: CAREER PATH RECOMMENDATION")
    print(Fore.CYAN + "=" * 80)

    path, match_pct = get_career_path_recommendation(valid_skills)

    print(f"\n🎯 Recommended Path: {Fore.YELLOW}{path}{Style.RESET_ALL}")
    print(f"   Skill Match: {match_pct:.1f}%")

    if "DevOps" in path:
        print("\n💼 DevOps/Infrastructure Specialist:")
        print("   • Average Salary: 829,397 KZT (+28.5% market premium)")
        print("   • Market Share: 27.4% of jobs")
        print("   • Key Skills: Docker, CI/CD, PostgreSQL, Git, Redis, Kubernetes")
        print("   • Career Focus: Infrastructure, automation, system reliability")

        print("\n📚 Skills You Have:")
        devops_user = [s for s in valid_skills if s in DEVOPS_SKILLS]
        if devops_user:
            print(f"   ✓ {', '.join(devops_user)}")

        print("\n🎓 Skills to Master:")
        devops_missing = [s for s in DEVOPS_SKILLS if s not in valid_skills]
        if devops_missing:
            print(f"   → {', '.join(devops_missing)}")
    else:
        print("\n💼 General Software Developer:")
        print("   • Average Salary: 645,644 KZT (market baseline)")
        print("   • Market Share: 72.6% of jobs")
        print("   • Key Skills: Git, Linux, JavaScript, Python, Agile, React")
        print("   • Career Focus: Application development, broad opportunities")

        print("\n📚 Skills You Have:")
        general_user = [s for s in valid_skills if s in GENERAL_DEV_SKILLS]
        if general_user:
            print(f"   ✓ {', '.join(general_user)}")

        print("\n🎓 Skills to Master:")
        general_missing = [s for s in GENERAL_DEV_SKILLS if s not in valid_skills]
        if general_missing:
            print(f"   → {', '.join(general_missing)}")

    # ================================================================
    # STEP 5: SKILL RECOMMENDATIONS
    # ================================================================

    print("\n" + Fore.CYAN + "=" * 80)
    print(Fore.CYAN + "STEP 5: HIGH-VALUE SKILLS TO LEARN NEXT")
    print(Fore.CYAN + "=" * 80)

    # Get skills user doesn't have
    user_skill_names = [s.replace('_', ' ').title() for s in valid_skills]
    skills_to_learn = skill_premiums[~skill_premiums['Skill'].isin(user_skill_names)].copy()
    skills_to_learn = skills_to_learn.sort_values('Coefficient', ascending=False).head(15)

    print("\n🚀 Top 15 Skills by Salary Impact (You Don't Have Yet):\n")
    print(f"{'Rank':<6} {'Skill':<20} {'Premium':>15} {'Frequency':>12} {'Status':>15}")
    print("-" * 70)

    for i, (_, row) in enumerate(skills_to_learn.iterrows(), 1):
        skill_name = row['Skill']
        premium = row['Coefficient']
        frequency = calculate_skill_frequency(skill_name)
        status = get_skill_gap_status(skill_name)

        # Color code by premium
        if premium >= 200000:
            color = Fore.GREEN
            emoji = "🔥"
        elif premium >= 100000:
            color = Fore.YELLOW
            emoji = "⭐"
        else:
            color = Fore.WHITE
            emoji = "📌"

        print(
            f"{i:<6} {skill_name:<20} {color}+{format_salary(premium):>14s} KZT{Style.RESET_ALL} {frequency:>11.1f}% {status:>15}")

    # ================================================================
    # STEP 6: KBTU GAP ANALYSIS
    # ================================================================

    print("\n" + Fore.CYAN + "=" * 80)
    print(Fore.CYAN + "STEP 6: KBTU CURRICULUM GAP - WHAT TO SELF-STUDY")
    print(Fore.CYAN + "=" * 80)

    # Identify gaps in user's skills
    user_gaps = [s for s in CRITICAL_GAPS if s not in valid_skills]

    if user_gaps:
        print("\n🔴 CRITICAL GAPS (Not taught at KBTU, high market demand):\n")

        for gap_skill in user_gaps:
            gap_skill_title = gap_skill.replace('_', ' ').title()
            gap_info = skill_premiums[skill_premiums['Skill'] == gap_skill_title]

            if len(gap_info) > 0:
                premium = gap_info.iloc[0]['Coefficient']
                frequency = calculate_skill_frequency(gap_skill_title)

                print(f"   📚 {gap_skill.upper()}:")
                print(f"      • Premium: +{format_salary(premium)} KZT")
                print(f"      • Market Demand: {frequency:.1f}% of jobs")

                # Learning resources
                if gap_skill == 'redis':
                    print(f"      • Learn: Redis University (free) - 20 hours")
                    print(f"      • Link: https://university.redis.com/")
                elif gap_skill == 'mongodb':
                    print(f"      • Learn: MongoDB University (free) - 30 hours")
                    print(f"      • Link: https://learn.mongodb.com/")
                elif gap_skill == 'elasticsearch':
                    print(f"      • Learn: Elastic Training (free tier) - 40 hours")
                    print(f"      • Link: https://www.elastic.co/training/")
                print()
    else:
        print(Fore.GREEN + "\n✓ Great! You've covered all critical gaps.")

    # KBTU skills not yet mastered
    kbtu_to_learn = []
    for s in KBTU_TAUGHT:
        if s not in valid_skills:
            skill_title = s.replace('_', ' ').title()
            if skill_title in skill_premiums['Skill'].values:
                kbtu_to_learn.append(skill_title)

    if kbtu_to_learn and len(kbtu_to_learn) <= 10:
        print("\n✅ KBTU-TAUGHT SKILLS YOU SHOULD MASTER:\n")
        print("   (These are in your curriculum - leverage them!)\n")

        for skill_title in kbtu_to_learn[:10]:
            skill_info = skill_premiums[skill_premiums['Skill'] == skill_title]
            if len(skill_info) > 0:
                premium = skill_info.iloc[0]['Coefficient']
                print(f"   • {skill_title:<20} +{format_salary(premium):>14s} KZT")

    # ================================================================
    # STEP 7: ACTION PLAN
    # ================================================================

    print("\n" + Fore.CYAN + "=" * 80)
    print(Fore.CYAN + "STEP 7: YOUR PERSONALIZED ACTION PLAN")
    print(Fore.CYAN + "=" * 80)

    print("\n🎯 IMMEDIATE ACTIONS (Next 1-3 months):\n")

    # Top 3 recommendations
    top_3_skills = skills_to_learn.head(3)

    for i, (_, row) in enumerate(top_3_skills.iterrows(), 1):
        skill_name = row['Skill']
        premium = row['Coefficient']
        status = get_skill_gap_status(skill_name)

        print(f"   {i}. Learn {Fore.YELLOW}{skill_name.upper()}{Style.RESET_ALL}")
        print(f"      • Impact: +{format_salary(premium)} KZT salary increase")
        print(f"      • Status: {status}")

        # Estimate learning time
        skill_lower = skill_name.lower().replace(' ', '_')
        if skill_lower in ['redis', 'mongodb', 'elasticsearch']:
            hours = {'redis': 20, 'mongodb': 30, 'elasticsearch': 40}.get(skill_lower, 30)
            print(f"      • Time: ~{hours} hours (self-study)")
        elif skill_lower in ['docker', 'kubernetes', 'ci/cd']:
            print(f"      • Time: ~40-60 hours (online courses + projects)")
        else:
            print(f"      • Time: ~30-50 hours (varies by background)")
        print()

    print("\n📚 MEDIUM-TERM GOALS (3-6 months):\n")
    print(f"   • Master your career path ({path})")
    print(f"   • Build portfolio projects showcasing top skills")
    print(f"   • Target salary: {Fore.GREEN}{format_salary(predicted_salary * 1.2)} KZT{Style.RESET_ALL} (+20%)")

    print("\n🚀 LONG-TERM VISION (6-12 months):\n")
    print(f"   • Achieve top 25% salary range: {Fore.GREEN}{format_salary(market_stats['q75'])} KZT{Style.RESET_ALL}")
    print(f"   • Specialize in high-premium niche")
    print(f"   • Consider senior/lead positions")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================

    print("\n" + Fore.CYAN + "=" * 80)
    print(Fore.CYAN + "SUMMARY")
    print(Fore.CYAN + "=" * 80)

    print(f"\n📊 Current Status:")
    print(f"   • Skills: {len(valid_skills)} recognized")
    print(f"   • Predicted Salary: {Fore.YELLOW}{format_salary(predicted_salary)} KZT{Style.RESET_ALL}")
    print(f"   • Market Percentile: {percentile:.1f}th")
    print(f"   • Career Path: {path}")

    print(f"\n🎯 Next Steps:")
    print(f"   1. Learn {top_3_skills.iloc[0]['Skill']} (+{format_salary(top_3_skills.iloc[0]['Coefficient'])} KZT)")
    if len(user_gaps) > 0:
        print(f"   2. Close KBTU gap: {user_gaps[0]} (self-study)")
    print(f"   3. Build portfolio project demonstrating skills")

    print(f"\n💡 Remember:")
    print(f"   • KBTU gives you excellent foundation (83% market coverage)")
    print(f"   • Association rules show GitLab CI + Docker + CI/CD bundled (Lift 10.68)")
    print(f"   • DevOps path offers 28.5% salary premium but fewer jobs")

    print("\n" + Fore.GREEN + "=" * 80)
    print(Fore.GREEN + "  GOOD LUCK WITH YOUR CAREER! 🌟")
    print(Fore.GREEN + "=" * 80 + "\n")


# ================================================================
# RUN THE ADVISOR
# ================================================================

if __name__ == "__main__":
    try:
        run_career_advisor()
    except KeyboardInterrupt:
        print("\n\n" + Fore.YELLOW + "Career advisor interrupted. Goodbye!")
    except Exception as e:
        print(Fore.RED + f"\n❌ An error occurred: {str(e)}")
        import traceback

        traceback.print_exc()
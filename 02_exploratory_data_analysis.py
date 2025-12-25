import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*70)
print("  EXPLORATORY DATA ANALYSIS (EDA)")
print("  Kazakhstan Tech Labor Market")
print("="*70)

# Load processed data
df = pd.read_csv('data_processed.csv')

print(f"\n📊 Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# SECTION 1: SALARY DISTRIBUTION ANALYSIS

def analyze_salary_distribution():
    """Analyze salary distributions across different dimensions"""
    print("\n" + "="*70)
    print("SECTION 1: SALARY DISTRIBUTION ANALYSIS")
    print("="*70)
    
    salaries = df['salary_monthly_kzt'].dropna()
    
    # Basic statistics
    print("\n📊 SALARY STATISTICS (KZT/month):")
    print("-" * 70)
    print(f"   Count:      {len(salaries):,}")
    print(f"   Mean:       {salaries.mean():,.0f} KZT")
    print(f"   Median:     {salaries.median():,.0f} KZT")
    print(f"   Std Dev:    {salaries.std():,.0f} KZT")
    print(f"   Min:        {salaries.min():,.0f} KZT")
    print(f"   25%:        {salaries.quantile(0.25):,.0f} KZT")
    print(f"   50%:        {salaries.quantile(0.50):,.0f} KZT")
    print(f"   75%:        {salaries.quantile(0.75):,.0f} KZT")
    print(f"   Max:        {salaries.max():,.0f} KZT")
    
    # Test for normality
    _, p_value = stats.shapiro(salaries.sample(min(5000, len(salaries))))
    print(f"\n   📈 Shapiro-Wilk normality test:")
    print(f"      p-value: {p_value:.4f}")
    print(f"      Distribution is {'NORMAL' if p_value > 0.05 else 'NOT NORMAL'} (α=0.05)")
    
    # Skewness and Kurtosis
    print(f"\n   📉 Distribution shape:")
    print(f"      Skewness: {stats.skew(salaries):.3f} ({'right-skewed' if stats.skew(salaries) > 0 else 'left-skewed'})")
    print(f"      Kurtosis: {stats.kurtosis(salaries):.3f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Salary Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Histogram with KDE
    axes[0, 0].hist(salaries, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(salaries.mean(), color='red', linestyle='--', label=f'Mean: {salaries.mean():,.0f}')
    axes[0, 0].axvline(salaries.median(), color='green', linestyle='--', label=f'Median: {salaries.median():,.0f}')
    axes[0, 0].set_xlabel('Salary (KZT/month)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Salary Distribution (Histogram)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot
    axes[0, 1].boxplot(salaries, vert=True)
    axes[0, 1].set_ylabel('Salary (KZT/month)')
    axes[0, 1].set_title('Salary Distribution (Box Plot)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(salaries, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normality Check)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # CDF
    sorted_salaries = np.sort(salaries)
    cdf = np.arange(1, len(sorted_salaries) + 1) / len(sorted_salaries)
    axes[1, 1].plot(sorted_salaries, cdf)
    axes[1, 1].set_xlabel('Salary (KZT/month)')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Distribution Function (CDF)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('01_salary_distribution.png', dpi=300, bbox_inches='tight')
    print("\n   💾 Visualization saved: 01_salary_distribution.png")
    plt.close()


# SECTION 2: SALARY BY LOCATION (Regional Analysis Preview)


def analyze_salary_by_location():
    """Compare salaries across different cities"""
    print("\n" + "="*70)
    print("SECTION 2: SALARY BY LOCATION")
    print("="*70)
    
    # Filter out rows without salary
    df_with_salary = df[df['salary_monthly_kzt'].notna()].copy()
    
    # Group by location
    location_stats = df_with_salary.groupby('area_normalized')['salary_monthly_kzt'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).round(0)
    
    print("\n📍 SALARY STATISTICS BY CITY:")
    print("-" * 70)
    print(location_stats.sort_values('mean', ascending=False).to_string())
    
    # Perform ANOVA test (preview for Problem 3)
    cities = df_with_salary['area_normalized'].unique()
    city_salaries = [df_with_salary[df_with_salary['area_normalized'] == city]['salary_monthly_kzt'].values 
                     for city in cities]
    
    f_stat, p_value = stats.f_oneway(*city_salaries)
    
    print(f"\n   📊 ANOVA Test (Are salary differences significant?):")
    print(f"      F-statistic: {f_stat:.3f}")
    print(f"      p-value: {p_value:.6f}")
    print(f"      Result: {'SIGNIFICANT differences' if p_value < 0.05 else 'NO significant differences'} (α=0.05)")
    
    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Salary Analysis by Location', fontsize=16, fontweight='bold')
    
    # Box plot
    df_with_salary.boxplot(column='salary_monthly_kzt', by='area_normalized', ax=axes[0])
    axes[0].set_title('Salary Distribution by City')
    axes[0].set_xlabel('City')
    axes[0].set_ylabel('Salary (KZT/month)')
    axes[0].tick_params(axis='x', rotation=45)
    plt.sca(axes[0])
    plt.xticks(rotation=45, ha='right')
    
    # Bar plot with error bars
    location_stats_plot = location_stats.sort_values('mean', ascending=False)
    x_pos = np.arange(len(location_stats_plot))
    axes[1].bar(x_pos, location_stats_plot['mean'], yerr=location_stats_plot['std'], 
                capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(location_stats_plot.index, rotation=45, ha='right')
    axes[1].set_ylabel('Mean Salary (KZT/month)')
    axes[1].set_xlabel('City')
    axes[1].set_title('Average Salary by City (with Std Dev)')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('02_salary_by_location.png', dpi=300, bbox_inches='tight')
    print("\n   💾 Visualization saved: 02_salary_by_location.png")
    plt.close()


# SECTION 3: SALARY BY EXPERIENCE LEVEL

def analyze_salary_by_experience():
    """Analyze how salary varies with experience"""
    print("\n" + "="*70)
    print("SECTION 3: SALARY BY EXPERIENCE LEVEL")
    print("="*70)
    
    df_with_salary = df[df['salary_monthly_kzt'].notna()].copy()
    
    # Group by experience
    exp_mapping = {0: 'No Experience', 1: '1-3 years', 2: '3-6 years', 3: '6+ years'}
    df_with_salary['experience_label'] = df_with_salary['experience_level'].map(exp_mapping)
    
    exp_stats = df_with_salary.groupby('experience_label')['salary_monthly_kzt'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std')
    ]).round(0)
    
    print("\n💼 SALARY BY EXPERIENCE:")
    print("-" * 70)
    print(exp_stats.to_string())
    
    # Calculate salary growth
    print("\n   📈 SALARY GROWTH:")
    exp_means = exp_stats['mean'].to_dict()
    base_salary = exp_means.get('No Experience', 0)
    
    for exp, salary in exp_means.items():
        if exp != 'No Experience' and base_salary > 0:
            growth_pct = ((salary - base_salary) / base_salary) * 100
            print(f"      {exp:20s}: +{growth_pct:6.1f}% vs. Entry-level")
    
    # Correlation test
    correlation, p_value = pearsonr(df_with_salary['experience_level'], 
                                    df_with_salary['salary_monthly_kzt'])
    
    print(f"\n   📊 Pearson Correlation (Experience vs Salary):")
    print(f"      Correlation coefficient: {correlation:.3f}")
    print(f"      p-value: {p_value:.6f}")
    print(f"      Result: {'SIGNIFICANT positive correlation' if p_value < 0.05 and correlation > 0 else 'No significant correlation'}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Salary vs Experience Analysis', fontsize=16, fontweight='bold')
    
    # Box plot
    order = ['No Experience', '1-3 years', '3-6 years', '6+ years']
    df_with_salary.boxplot(column='salary_monthly_kzt', by='experience_label', ax=axes[0])
    axes[0].set_title('Salary Distribution by Experience Level')
    axes[0].set_xlabel('Experience Level')
    axes[0].set_ylabel('Salary (KZT/month)')
    plt.sca(axes[0])
    plt.xticks(rotation=45, ha='right')
    
    # Scatter plot with trend line
    axes[1].scatter(df_with_salary['experience_level'], df_with_salary['salary_monthly_kzt'], 
                   alpha=0.3, s=20)
    
    # Add trend line
    z = np.polyfit(df_with_salary['experience_level'], df_with_salary['salary_monthly_kzt'], 1)
    p = np.poly1d(z)
    axes[1].plot(df_with_salary['experience_level'].unique(), 
                p(df_with_salary['experience_level'].unique()), 
                "r-", linewidth=2, label=f'Trend: y = {z[0]:.0f}x + {z[1]:.0f}')
    
    axes[1].set_xlabel('Experience Level (0=None, 3=6+ years)')
    axes[1].set_ylabel('Salary (KZT/month)')
    axes[1].set_title('Salary vs Experience (Scatter Plot)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('03_salary_by_experience.png', dpi=300, bbox_inches='tight')
    print("\n   💾 Visualization saved: 03_salary_by_experience.png")
    plt.close()


# SECTION 4: SKILL FREQUENCY ANALYSIS

def analyze_skill_frequency():
    """Analyze most demanded skills"""
    print("\n" + "="*70)
    print("SECTION 4: SKILL DEMAND ANALYSIS")
    print("="*70)
    
    # Get skill columns
    skill_cols = [col for col in df.columns if col.startswith('skill_')]
    
    # Calculate skill frequencies
    skill_freq = {}
    for col in skill_cols:
        skill_name = col.replace('skill_', '').replace('_', ' ').title()
        frequency = df[col].sum()
        percentage = (frequency / len(df)) * 100
        skill_freq[skill_name] = {
            'frequency': frequency,
            'percentage': percentage
        }
    
    # Convert to DataFrame and sort
    skill_df = pd.DataFrame(skill_freq).T
    skill_df = skill_df.sort_values('frequency', ascending=False).head(20)
    
    print("\n🔥 TOP 20 MOST DEMANDED SKILLS:")
    print("-" * 70)
    for idx, (skill, row) in enumerate(skill_df.iterrows(), 1):
        print(f"   {idx:2d}. {skill:25s}: {row['frequency']:4.0f} jobs ({row['percentage']:5.1f}%)")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Top 20 Most Demanded Skills', fontsize=16, fontweight='bold')
    
    # Horizontal bar chart
    axes[0].barh(range(len(skill_df)), skill_df['frequency'], color='skyblue', edgecolor='black')
    axes[0].set_yticks(range(len(skill_df)))
    axes[0].set_yticklabels(skill_df.index)
    axes[0].set_xlabel('Number of Job Postings')
    axes[0].set_title('Skill Frequency (Absolute)')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Percentage chart
    axes[1].barh(range(len(skill_df)), skill_df['percentage'], color='lightcoral', edgecolor='black')
    axes[1].set_yticks(range(len(skill_df)))
    axes[1].set_yticklabels(skill_df.index)
    axes[1].set_xlabel('Percentage of Jobs (%)')
    axes[1].set_title('Skill Demand (Percentage)')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('04_skill_frequency.png', dpi=300, bbox_inches='tight')
    print("\n   💾 Visualization saved: 04_skill_frequency.png")
    plt.close()


# SECTION 5: SKILL-SALARY CORRELATION

def analyze_skill_salary_correlation():
    """
    Analyze correlation between skills and salary
    Covers Lecture #3: Correlation Analysis
    """
    print("\n" + "="*70)
    print("SECTION 5: SKILL-SALARY CORRELATION")
    print("="*70)
    
    # Filter jobs with salary
    df_with_salary = df[df['salary_monthly_kzt'].notna()].copy()
    skill_cols = [col for col in df.columns if col.startswith('skill_')]
    
    # Calculate point-biserial correlation for each skill
    correlations = {}
    for col in skill_cols:
        skill_name = col.replace('skill_', '').replace('_', ' ').title()
        
        # Point-biserial correlation (binary variable vs continuous)
        has_skill = df_with_salary[col] == 1
        salary_with = df_with_salary[has_skill]['salary_monthly_kzt']
        salary_without = df_with_salary[~has_skill]['salary_monthly_kzt']
        
        if len(salary_with) > 10 and len(salary_without) > 10:
            # Calculate correlation
            correlation, p_value = stats.pointbiserialr(df_with_salary[col], 
                                                        df_with_salary['salary_monthly_kzt'])
            
            # Calculate mean difference
            mean_diff = salary_with.mean() - salary_without.mean()
            
            correlations[skill_name] = {
                'correlation': correlation,
                'p_value': p_value,
                'mean_with_skill': salary_with.mean(),
                'mean_without_skill': salary_without.mean(),
                'salary_premium': mean_diff,
                'count_with_skill': len(salary_with)
            }
    
    # Convert to DataFrame
    corr_df = pd.DataFrame(correlations).T
    corr_df = corr_df.sort_values('salary_premium', ascending=False).head(20)
    
    print("\n💎 TOP 20 SKILLS BY SALARY PREMIUM:")
    print("-" * 70)
    print(f"{'Skill':25s} {'Premium (KZT)':>15s} {'Correlation':>12s} {'P-value':>10s} {'Jobs':>8s}")
    print("-" * 70)
    
    for skill, row in corr_df.iterrows():
        significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"{skill:25s} {row['salary_premium']:>+14,.0f} {row['correlation']:>12.3f} {row['p_value']:>10.4f}{significance:3s} {row['count_with_skill']:>8.0f}")
    
    print("\n   *** p<0.001  ** p<0.01  * p<0.05")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Skill-Salary Correlation Analysis (Top 20)', fontsize=16, fontweight='bold')
    
    # Salary premium
    colors = ['green' if p < 0.05 else 'gray' for p in corr_df['p_value']]
    axes[0].barh(range(len(corr_df)), corr_df['salary_premium'], color=colors, edgecolor='black')
    axes[0].set_yticks(range(len(corr_df)))
    axes[0].set_yticklabels(corr_df.index)
    axes[0].set_xlabel('Salary Premium (KZT/month)')
    axes[0].set_title('Salary Premium per Skill (Green = Significant p<0.05)')
    axes[0].invert_yaxis()
    axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Correlation coefficients
    colors2 = ['green' if p < 0.05 else 'gray' for p in corr_df['p_value']]
    axes[1].barh(range(len(corr_df)), corr_df['correlation'], color=colors2, edgecolor='black')
    axes[1].set_yticks(range(len(corr_df)))
    axes[1].set_yticklabels(corr_df.index)
    axes[1].set_xlabel('Point-Biserial Correlation')
    axes[1].set_title('Skill-Salary Correlation Coefficient')
    axes[1].invert_yaxis()
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('05_skill_salary_correlation.png', dpi=300, bbox_inches='tight')
    print("\n   💾 Visualization saved: 05_skill_salary_correlation.png")
    plt.close()


# SECTION 6: CORRELATION MATRIX (ALL NUMERIC FEATURES)

def create_correlation_matrix():
    """Create correlation matrix for numeric features"""
    print("\n" + "="*70)
    print("SECTION 6: FEATURE CORRELATION MATRIX")
    print("="*70)
    
    # Select numeric features
    numeric_cols = ['salary_monthly_kzt', 'experience_level'] + \
                   [col for col in df.columns if col.startswith('skill_')][:20]  # Top 20 skills
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Visualization
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix: Salary, Experience & Top 20 Skills', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('06_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("\n   💾 Visualization saved: 06_correlation_matrix.png")
    plt.close()
    
    # Show strongest correlations with salary
    salary_corr = corr_matrix['salary_monthly_kzt'].sort_values(ascending=False).drop('salary_monthly_kzt')
    
    print("\n📊 STRONGEST CORRELATIONS WITH SALARY:")
    print("-" * 70)
    print(salary_corr.head(10).to_string())



if __name__ == "__main__":
    analyze_salary_distribution()
    analyze_salary_by_location()
    analyze_salary_by_experience()
    analyze_skill_frequency()
    analyze_skill_salary_correlation()
    create_correlation_matrix()
    
    print("\n" + "="*70)
    print("  EDA COMPLETED SUCCESSFULLY!")
    print("  6 visualizations saved")
    print("="*70)

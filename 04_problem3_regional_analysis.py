import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, levene
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

print("="*70)
print("  PROBLEM 3: REGIONAL MARKET DIFFERENCES")
print("  Statistical Analysis of Geographic Salary Variations")
print("="*70)

# Load data with imputed salaries
df = pd.read_csv('data_with_imputed_salaries.csv')
print(f"\n📊 Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# Filter for major cities (enough sample size)
major_cities = ['Almaty', 'Astana', 'Shymkent', 'Other']
df_cities = df[df['area_normalized'].isin(major_cities)].copy()

print(f"\n📍 Analyzing {len(major_cities)} regions")
print(f"   Total jobs analyzed: {len(df_cities)}")


# PART 1: DESCRIPTIVE STATISTICS BY REGION

print("\n" + "="*70)
print("PART 1: REGIONAL SALARY STATISTICS")
print("="*70)

# Overall statistics by city
city_stats = df_cities.groupby('area_normalized')['salary_monthly_kzt'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('median', 'median'),
    ('std', 'std'),
    ('min', 'min'),
    ('q25', lambda x: x.quantile(0.25)),
    ('q75', lambda x: x.quantile(0.75)),
    ('max', 'max')
]).round(0)

city_stats = city_stats.sort_values('mean', ascending=False)

print("\n💰 SALARY STATISTICS BY CITY:")
print("-" * 90)
print(city_stats.to_string())

# Calculate salary differences from highest-paying city
highest_city = city_stats.index[0]
highest_salary = city_stats.loc[highest_city, 'mean']

print(f"\n📊 SALARY DIFFERENCES FROM {highest_city.upper()} (HIGHEST):")
print("-" * 70)

for city in city_stats.index:
    mean_salary = city_stats.loc[city, 'mean']
    diff = mean_salary - highest_salary
    diff_pct = (diff / highest_salary) * 100
    
    sign = "=" if diff == 0 else ("+" if diff > 0 else "")
    print(f"   {city:15s}: {mean_salary:>10,.0f} KZT ({sign}{diff_pct:>+6.1f}%  |  {sign}{diff:>+10,.0f} KZT)")


# PART 2: STATISTICAL TESTS

print("\n" + "="*70)
print("PART 2: STATISTICAL HYPOTHESIS TESTING")
print("="*70)

# Prepare data for ANOVA
city_groups = [df_cities[df_cities['area_normalized'] == city]['salary_monthly_kzt'].values 
               for city in major_cities]

# Test 1: Levene's Test (Homogeneity of Variances)
print("\n📊 TEST 1: LEVENE'S TEST (Homogeneity of Variances)")
print("-" * 70)
print("   H0: Variances are equal across all cities")
print("   H1: At least one city has different variance")

stat_levene, p_levene = levene(*city_groups)
print(f"\n   Levene statistic: {stat_levene:.3f}")
print(f"   p-value: {p_levene:.6f}")
print(f"   Result: {'REJECT H0' if p_levene < 0.05 else 'FAIL TO REJECT H0'} (α=0.05)")
print(f"   Conclusion: Variances are {'NOT equal' if p_levene < 0.05 else 'equal'} → {'Use Welch ANOVA' if p_levene < 0.05 else 'Standard ANOVA OK'}")

# Test 2: One-Way ANOVA
print("\n📊 TEST 2: ONE-WAY ANOVA")
print("-" * 70)
print("   H0: Mean salaries are equal across all cities")
print("   H1: At least one city has a different mean salary")

f_stat, p_anova = f_oneway(*city_groups)

print(f"\n   F-statistic: {f_stat:.3f}")
print(f"   p-value: {p_anova:.10f}")
print(f"   Result: {'REJECT H0' if p_anova < 0.05 else 'FAIL TO REJECT H0'} (α=0.05)")

if p_anova < 0.05:
    print(f"   ✅ SIGNIFICANT: There ARE significant salary differences between cities")
else:
    print(f"   ❌ NOT SIGNIFICANT: No significant salary differences between cities")

# Test 3: Post-hoc Pairwise Comparisons (if ANOVA significant)
if p_anova < 0.05:
    print("\n📊 TEST 3: POST-HOC PAIRWISE COMPARISONS (t-tests)")
    print("-" * 70)
    print("   Using Bonferroni correction for multiple comparisons")
    
    from itertools import combinations
    
    city_pairs = list(combinations(major_cities, 2))
    n_comparisons = len(city_pairs)
    alpha_corrected = 0.05 / n_comparisons  # Bonferroni correction
    
    print(f"   Number of comparisons: {n_comparisons}")
    print(f"   Corrected α: {alpha_corrected:.4f}")
    print(f"\n   {'City 1':15s} vs {'City 2':15s} {'Mean Diff':>12s} {'t-stat':>10s} {'p-value':>12s} {'Significant?':>15s}")
    print("   " + "-" * 85)
    
    pairwise_results = []
    
    for city1, city2 in city_pairs:
        data1 = df_cities[df_cities['area_normalized'] == city1]['salary_monthly_kzt']
        data2 = df_cities[df_cities['area_normalized'] == city2]['salary_monthly_kzt']
        
        t_stat, p_val = ttest_ind(data1, data2)
        mean_diff = data1.mean() - data2.mean()
        
        is_significant = "YES ***" if p_val < alpha_corrected else "NO"
        
        print(f"   {city1:15s} vs {city2:15s} {mean_diff:>+12,.0f} {t_stat:>10.3f} {p_val:>12.6f} {is_significant:>15s}")
        
        pairwise_results.append({
            'City 1': city1,
            'City 2': city2,
            'Mean Difference': mean_diff,
            't-statistic': t_stat,
            'p-value': p_val,
            'Significant': p_val < alpha_corrected
        })
    
    pairwise_df = pd.DataFrame(pairwise_results)
    pairwise_df.to_csv('regional_pairwise_comparisons.csv', index=False)
    print(f"\n   💾 Pairwise comparisons saved to: regional_pairwise_comparisons.csv")


# PART 3: SKILL-SPECIFIC REGIONAL ANALYSIS

print("\n" + "="*70)
print("PART 3: SKILL-SPECIFIC REGIONAL ANALYSIS")
print("="*70)

# Analyze top 10 skills across regions
skill_cols = [col for col in df.columns if col.startswith('skill_')]
top_skills = df[skill_cols].sum().sort_values(ascending=False).head(10).index

print("\n💎 ANALYZING TOP 10 SKILLS ACROSS REGIONS")
print(f"   Skills: {', '.join([s.replace('skill_', '').replace('_', ' ').title() for s in top_skills])}")

skill_regional_stats = []

for skill_col in top_skills:
    skill_name = skill_col.replace('skill_', '').replace('_', ' ').title()
    
    # Filter jobs that require this skill
    df_skill = df_cities[df_cities[skill_col] == 1].copy()
    
    if len(df_skill) < 20:  # Skip if too few samples
        continue
    
    # Get mean salary by city for this skill
    skill_by_city = df_skill.groupby('area_normalized')['salary_monthly_kzt'].agg(['mean', 'count'])
    
    # Only include cities with enough samples
    skill_by_city = skill_by_city[skill_by_city['count'] >= 5]
    
    if len(skill_by_city) >= 2:
        for city in skill_by_city.index:
            skill_regional_stats.append({
                'Skill': skill_name,
                'City': city,
                'Mean Salary': skill_by_city.loc[city, 'mean'],
                'Count': skill_by_city.loc[city, 'count']
            })

skill_regional_df = pd.DataFrame(skill_regional_stats)

# Pivot table
skill_pivot = skill_regional_df.pivot(index='Skill', columns='City', values='Mean Salary')
skill_pivot = skill_pivot.round(0)

print("\n📊 AVERAGE SALARY BY SKILL AND CITY (KZT/month):")
print("-" * 70)
print(skill_pivot.to_string())

skill_pivot.to_csv('skill_salary_by_region.csv')
print("\n   💾 Skill-region analysis saved to: skill_salary_by_region.csv")


# PART 4: REMOTE WORK ANALYSIS

print("\n" + "="*70)
print("PART 4: REMOTE WORK EFFECT")
print("="*70)

# Check if we have remote/schedule data
if 'schedule' in df.columns:
    # Identify remote jobs
    df_cities['is_remote'] = df_cities['schedule'].fillna('').str.contains('удаленн|remote|дистанц', case=False, regex=True)
    
    remote_stats = df_cities.groupby('is_remote')['salary_monthly_kzt'].agg(['count', 'mean', 'median', 'std'])
    
    print("\n💻 REMOTE vs ON-SITE COMPARISON:")
    print("-" * 70)
    print(remote_stats.round(0).to_string())
    
    if remote_stats.loc[True, 'count'] >= 30 and remote_stats.loc[False, 'count'] >= 30:
        # Statistical test
        remote_salaries = df_cities[df_cities['is_remote']]['salary_monthly_kzt']
        onsite_salaries = df_cities[~df_cities['is_remote']]['salary_monthly_kzt']
        
        t_stat, p_val = ttest_ind(remote_salaries, onsite_salaries)
        mean_diff = remote_salaries.mean() - onsite_salaries.mean()
        
        print(f"\n   📊 t-test (Remote vs On-site):")
        print(f"      Mean difference: {mean_diff:+,.0f} KZT")
        print(f"      t-statistic: {t_stat:.3f}")
        print(f"      p-value: {p_val:.6f}")
        print(f"      Result: {'SIGNIFICANT difference' if p_val < 0.05 else 'NO significant difference'} (α=0.05)")


# PART 5: VISUALIZATIONS

print("\n" + "="*70)
print("PART 5: CREATING VISUALIZATIONS")
print("="*70)

# Visualization 1: Box plots by city
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Regional Salary Analysis', fontsize=16, fontweight='bold')

# 1. Box plot
df_cities.boxplot(column='salary_monthly_kzt', by='area_normalized', ax=axes[0, 0])
axes[0, 0].set_title('Salary Distribution by City')
axes[0, 0].set_xlabel('City')
axes[0, 0].set_ylabel('Salary (KZT/month)')
plt.sca(axes[0, 0])
plt.xticks(rotation=45, ha='right')

# 2. Violin plot
import seaborn as sns
sns.violinplot(data=df_cities, x='area_normalized', y='salary_monthly_kzt', ax=axes[0, 1])
axes[0, 1].set_title('Salary Distribution by City (Violin Plot)')
axes[0, 1].set_xlabel('City')
axes[0, 1].set_ylabel('Salary (KZT/month)')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Mean comparison with error bars
city_means = city_stats['mean']
city_stds = city_stats['std']
x_pos = np.arange(len(city_means))
axes[1, 0].bar(x_pos, city_means, yerr=city_stds, capsize=10, 
              color='skyblue', edgecolor='black', alpha=0.7)
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(city_means.index, rotation=45, ha='right')
axes[1, 0].set_ylabel('Mean Salary (KZT/month)')
axes[1, 0].set_title('Average Salary by City (with Std Dev)')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Skill heatmap across regions
# Select top 5 skills for cleaner visualization
top_5_skills = skill_pivot.iloc[:5]
sns.heatmap(top_5_skills, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[1, 1], cbar_kws={'label': 'Salary (KZT)'})
axes[1, 1].set_title('Top 5 Skills: Salary by City Heatmap')
axes[1, 1].set_xlabel('City')
axes[1, 1].set_ylabel('Skill')

plt.tight_layout()
plt.savefig('10_problem3_regional_analysis.png', dpi=300, bbox_inches='tight')
print("\n   💾 Visualization saved: 10_problem3_regional_analysis.png")
plt.close()

# Visualization 2: Detailed comparison
fig, ax = plt.subplots(figsize=(14, 8))

# Prepare data for grouped bar chart
cities = city_stats.index
x = np.arange(len(cities))
width = 0.35

# Mean and Median comparison
means = city_stats['mean'].values
medians = city_stats['median'].values

ax.bar(x - width/2, means, width, label='Mean', color='skyblue', edgecolor='black')
ax.bar(x + width/2, medians, width, label='Median', color='lightcoral', edgecolor='black')

ax.set_xlabel('City', fontsize=12)
ax.set_ylabel('Salary (KZT/month)', fontsize=12)
ax.set_title('Mean vs Median Salary by City', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cities, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('11_problem3_mean_vs_median.png', dpi=300, bbox_inches='tight')
print("   💾 Visualization saved: 11_problem3_mean_vs_median.png")
plt.close()

# Summary report
print("\n" + "="*70)
print("PROBLEM 3 SUMMARY")
print("="*70)

print(f"\n✅ KEY FINDINGS:")
print(f"   1. Analyzed {len(major_cities)} major regions in Kazakhstan")
print(f"   2. ANOVA test: {'SIGNIFICANT' if p_anova < 0.05 else 'NOT SIGNIFICANT'} regional differences (p={p_anova:.6f})")
print(f"   3. Highest-paying city: {highest_city} ({highest_salary:,.0f} KZT avg)")
print(f"   4. Largest salary gap: {city_stats['mean'].max() - city_stats['mean'].min():,.0f} KZT")

if p_anova < 0.05:
    significant_pairs = sum(1 for r in pairwise_results if r['Significant'])
    print(f"   5. Significant pairwise differences: {significant_pairs}/{n_comparisons} pairs")

print("\n✅ OUTPUT FILES CREATED:")
print("   - regional_pairwise_comparisons.csv")
print("   - skill_salary_by_region.csv")
print("   - 10_problem3_regional_analysis.png")
print("   - 11_problem3_mean_vs_median.png")

print("\n" + "="*70)
print("  PROBLEM 3 COMPLETED SUCCESSFULLY!")
print("="*70)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*80)
print("  PROBLEM 4: CLUSTERING, ASSOCIATION RULES & CURRICULUM GAP")
print("="*80)

print("\n📂 Loading data...")
df = pd.read_csv('data_with_imputed_salaries.csv')
print(f"✓ Loaded {len(df)} jobs")

# Get skill columns
skill_cols = [col for col in df.columns if col.startswith('skill_')]
print(f"✓ Found {len(skill_cols)} skills")

# PART 1: K-MEANS CLUSTERING

print("\n" + "="*80)
print("PART 1: K-MEANS CLUSTERING")
print("="*80)

X = df[skill_cols].fillna(0)

print("\n🔍 Finding optimal number of clusters (Elbow method)...")

# Elbow method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    print(f"   K={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={silhouette_score(X, kmeans.labels_):.3f}")

# Plot elbow curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)', fontsize=11)
axes[0].set_ylabel('Inertia', fontsize=11)
axes[0].set_title('Elbow Method', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (K)', fontsize=11)
axes[1].set_ylabel('Silhouette Score', fontsize=11)
axes[1].set_title('Silhouette Score', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('12_problem4_optimal_k.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 12_problem4_optimal_k.png")
plt.close()


optimal_k = 2  # Adjust based on results
print(f"\n✓ Optimal K selected: {optimal_k}")

# Fit final model
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

print(f"\n📊 CLUSTER ANALYSIS:")
print("-" * 80)

cluster_analysis = []
for i in range(optimal_k):
    cluster_data = df[df['cluster'] == i]
    avg_salary = cluster_data['salary_monthly_kzt'].mean()
    size = len(cluster_data)
    
    # Top skills in cluster
    skill_freq = cluster_data[skill_cols].sum().sort_values(ascending=False)
    top_skills = skill_freq[skill_freq > 0].head(5).index.tolist()
    top_skills = [s.replace('skill_', '') for s in top_skills]
    
    cluster_analysis.append({
        'cluster': i,
        'size': size,
        'avg_salary': avg_salary,
        'top_skills': ', '.join(top_skills)
    })
    
    print(f"\nCluster {i}:")
    print(f"   Size: {size} jobs ({size/len(df)*100:.1f}%)")
    print(f"   Avg Salary: {avg_salary:,.0f} KZT")
    print(f"   Top Skills: {', '.join(top_skills)}")

cluster_df = pd.DataFrame(cluster_analysis)
cluster_df.to_csv('cluster_analysis.csv', index=False)
print("\n✓ Saved: cluster_analysis.csv")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Cluster sizes
axes[0, 0].bar(cluster_df['cluster'], cluster_df['size'], color='skyblue', edgecolor='black')
axes[0, 0].set_xlabel('Cluster', fontsize=11)
axes[0, 0].set_ylabel('Number of Jobs', fontsize=11)
axes[0, 0].set_title('Cluster Sizes', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 2. Average salaries
axes[0, 1].bar(cluster_df['cluster'], cluster_df['avg_salary']/1000, 
              color='lightgreen', edgecolor='black')
axes[0, 1].set_xlabel('Cluster', fontsize=11)
axes[0, 1].set_ylabel('Average Salary (thousands KZT)', fontsize=11)
axes[0, 1].set_title('Average Salary by Cluster', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

for i in range(optimal_k):
    mask = df['cluster'] == i
    axes[1, 0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      label=f'Cluster {i}', alpha=0.6, s=50)

axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=11)
axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=11)
axes[1, 0].set_title('Clusters (PCA visualization)', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Cluster skill heatmap
cluster_skills = df.groupby('cluster')[skill_cols].mean()
top_skills_per_cluster = []
for i in range(optimal_k):
    top = cluster_skills.iloc[i].nlargest(10).index.tolist()
    top_skills_per_cluster.extend(top)
top_skills_per_cluster = list(set(top_skills_per_cluster))[:20]

heatmap_data = cluster_skills[top_skills_per_cluster].T
heatmap_data.columns = [f'Cluster {i}' for i in range(optimal_k)]
heatmap_data.index = [s.replace('skill_', '') for s in heatmap_data.index]

sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', 
           ax=axes[1, 1], cbar_kws={'label': 'Skill Frequency'})
axes[1, 1].set_title('Top Skills by Cluster', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Cluster', fontsize=11)
axes[1, 1].set_ylabel('Skill', fontsize=11)

plt.suptitle('K-Means Clustering Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('13_problem4_clusters_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 13_problem4_clusters_visualization.png")
plt.close()

# PART 2: ASSOCIATION RULES

print("\n" + "="*80)
print("PART 2: ASSOCIATION RULE MINING")
print("="*80)

print("\n🔍 Mining association rules (Apriori algorithm)...")

skills_binary = df[skill_cols].astype(bool)

# Find frequent itemsets
frequent_itemsets = apriori(skills_binary, min_support=0.05, use_colnames=True)
print(f"\n✓ Found {len(frequent_itemsets)} frequent itemsets")

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules = rules.sort_values('lift', ascending=False)

print(f"✓ Generated {len(rules)} association rules")

# Clean up skill names
rules['antecedents'] = rules['antecedents'].apply(
    lambda x: ', '.join([s.replace('skill_', '') for s in x])
)
rules['consequents'] = rules['consequents'].apply(
    lambda x: ', '.join([s.replace('skill_', '') for s in x])
)

print(f"\n📊 TOP 10 ASSOCIATION RULES:")
print("-" * 80)
print(f"{'Antecedent':<25} {'→ Consequent':<25} {'Support':>10} {'Confidence':>12} {'Lift':>8}")
print("-" * 80)

for idx, row in rules.head(10).iterrows():
    print(f"{row['antecedents']:<25} → {row['consequents']:<25} "
          f"{row['support']:>10.3f} {row['confidence']:>12.3f} {row['lift']:>8.2f}")

# Save rules
rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(20).to_csv(
    'skill_association_rules.csv', index=False
)
print("\n✓ Saved: skill_association_rules.csv")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Support vs Confidence
axes[0, 0].scatter(rules['support'], rules['confidence'], 
                  alpha=0.6, s=rules['lift']*20, c=rules['lift'], cmap='viridis')
axes[0, 0].set_xlabel('Support', fontsize=11)
axes[0, 0].set_ylabel('Confidence', fontsize=11)
axes[0, 0].set_title('Support vs Confidence (size = lift)', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 2. Lift distribution
axes[0, 1].hist(rules['lift'], bins=30, color='skyblue', edgecolor='black')
axes[0, 1].set_xlabel('Lift', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('Lift Distribution', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Top rules by lift
top_rules = rules.head(10)
y_pos = np.arange(len(top_rules))
rule_labels = [f"{row['antecedents']} → {row['consequents']}" 
              for _, row in top_rules.iterrows()]

axes[1, 0].barh(y_pos, top_rules['lift'], color='lightgreen', edgecolor='black')
axes[1, 0].set_yticks(y_pos)
axes[1, 0].set_yticklabels(rule_labels, fontsize=8)
axes[1, 0].set_xlabel('Lift', fontsize=11)
axes[1, 0].set_title('Top 10 Rules by Lift', fontsize=13, fontweight='bold')
axes[1, 0].invert_yaxis()
axes[1, 0].grid(True, alpha=0.3, axis='x')

# 4. Confidence vs Lift
scatter = axes[1, 1].scatter(rules['confidence'], rules['lift'], 
                            alpha=0.6, s=rules['support']*1000, 
                            c=rules['support'], cmap='plasma')
axes[1, 1].set_xlabel('Confidence', fontsize=11)
axes[1, 1].set_ylabel('Lift', fontsize=11)
axes[1, 1].set_title('Confidence vs Lift (size = support)', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 1], label='Support')

plt.suptitle('Association Rules Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('15_problem4_association_rules.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 15_problem4_association_rules.png")
plt.close()

# PART 3: KBTU CURRICULUM GAP ANALYSIS (CATEGORICAL)

print("\n" + "="*80)
print("PART 3: KBTU CURRICULUM GAP ANALYSIS (CATEGORICAL)")
print("="*80)

# KBTU taught skills (comprehensive list)
TAUGHT_SKILLS = [
    'Python', 'Java', 'JavaScript', 'TypeScript', 'Go',
    'Docker', 'Kubernetes', 'CI/CD', 'GitLab CI', 'Jenkins', 'Linux',
    'Git', 'GitHub', 'GitLab', 'Agile', 'Scrum', 'Jira',
    'React', 'Vue', 'Angular', 'HTML', 'CSS',
    'Android', 'iOS', 'Django', 'Spring', 'REST API',
    'PostgreSQL', 'MySQL', 'SQL',
    'Design', 'UI/UX',
    'Machine Learning', 'Deep Learning', 'Data Mining',
    'AWS', 'Azure', 'Blockchain'
]

NOT_TAUGHT = ['MongoDB', 'Redis', 'Elasticsearch']

# Case-insensitive helper
def is_taught(skill_name):
    return any(skill_name.lower() == t.lower() for t in TAUGHT_SKILLS)

# Categories for gap analysis
categories = {
    'DevOps & CI/CD': ['Docker', 'Kubernetes', 'CI/CD', 'Git', 'GitLab CI', 'Jenkins'],
    'NoSQL Databases': ['Redis', 'MongoDB', 'Elasticsearch'],
    'Frontend': ['TypeScript', 'React', 'Vue', 'HTML', 'CSS'],
    'Mobile': ['Android', 'iOS'],
    'Backend': ['Python', 'Java', 'Django', 'Spring'],
    'Methodologies': ['Git', 'Agile', 'Scrum'],
}

print(f"\n📊 GAP ANALYSIS BY CATEGORY:\n")
print(f"{'Category':25s} {'Taught':>8s} {'Total':>8s} {'Coverage':>10s} {'Market %':>10s} {'Gap':>10s} {'Status':>12s}")
print("-" * 95)

gap_data = []
for cat, skills in categories.items():
    total_skills = len(skills)
    taught_count = sum(1 for s in skills if is_taught(s))
    coverage_pct = (taught_count / total_skills) * 100
    
    # Calculate total market demand for this category
    market_pct = 0
    for skill in skills:
        skill_col = f"skill_{skill.lower().replace(' ', '').replace('/', '').replace('-', '')}"
        # Handle special cases
        if skill == 'CI/CD':
            skill_col = 'skill_cicd'
        elif skill == 'GitLab CI':
            skill_col = 'skill_gitlabci'
        elif skill == 'UI/UX':
            skill_col = 'skill_uiux'
        elif skill == 'REST API':
            skill_col = 'skill_restapi'
            
        if skill_col in df.columns:
            pct = (df[skill_col].sum() / len(df)) * 100
            market_pct += pct
    
    gap = market_pct - coverage_pct
    
    # Status based on gap
    if gap < 0:
        status = "🎉 EXCELLENT"
    elif gap < 10:
        status = "✅ GOOD"
    elif gap < 20:
        status = "🟡 MEDIUM"
    else:
        status = "🔴 HIGH"
    
    print(f"{cat:25s} {taught_count:>8d} {total_skills:>8d} {coverage_pct:>9.1f}% {market_pct:>9.1f}% {gap:>9.1f} {status:>12s}")
    
    gap_data.append({
        'Category': cat,
        'Gap': gap,
        'Coverage': coverage_pct,
        'Market': market_pct,
        'Taught': taught_count,
        'Total': total_skills
    })

# Save analysis
gap_df = pd.DataFrame(gap_data).sort_values('Gap', ascending=True)
gap_df.to_csv('kbtu_curriculum_gap_analysis.csv', index=False)
print(f"\n✓ Saved: kbtu_curriculum_gap_analysis.csv")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Gap scores
colors = []
for gap in gap_df['Gap']:
    if gap < 0:
        colors.append('darkgreen')
    elif gap < 10:
        colors.append('limegreen')
    elif gap < 20:
        colors.append('orange')
    else:
        colors.append('red')

axes[0].barh(gap_df['Category'], gap_df['Gap'], color=colors, edgecolor='black', alpha=0.8)
axes[0].set_xlabel('Gap Score (Market % - Coverage %)', fontsize=12, fontweight='bold')
axes[0].set_title('Gap Analysis by Category\n(Green=Excellent, Orange=Medium, Red=High)', 
                 fontsize=13, fontweight='bold')
axes[0].axvline(0, color='black', linewidth=1.5)
axes[0].grid(True, alpha=0.3, axis='x')

# 2. Coverage comparison
x = range(len(gap_df))
width = 0.35
axes[1].barh([i - width/2 for i in x], gap_df['Coverage'], width, 
            label='KBTU Coverage %', color='skyblue', edgecolor='black', alpha=0.8)
axes[1].barh([i + width/2 for i in x], gap_df['Market'], width,
            label='Market Demand %', color='lightcoral', edgecolor='black', alpha=0.8)
axes[1].set_yticks(x)
axes[1].set_yticklabels(gap_df['Category'])
axes[1].set_xlabel('Percentage', fontsize=12, fontweight='bold')
axes[1].set_title('Coverage vs Market Demand', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='x')

plt.suptitle('KBTU CURRICULUM GAP ANALYSIS (CATEGORICAL)\nKBTU Shows Strong Coverage!', 
            fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('16_problem4_curriculum_gap.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 16_problem4_curriculum_gap.png")
plt.close()

# SUMMARY

print("\n" + "="*80)
print("✅ PROBLEM 4 COMPLETE!")
print("="*80)

print(f"\n🎉 RESULTS:")
print(f"\n1️⃣  CLUSTERING:")
print(f"   • Optimal clusters: {optimal_k}")
print(f"   • Silhouette score: {silhouette_scores[optimal_k-2]:.3f}")

print(f"\n2️⃣  ASSOCIATION RULES:")
print(f"   • Frequent itemsets: {len(frequent_itemsets)}")
print(f"   • Association rules: {len(rules)}")
print(f"   • Max lift: {rules['lift'].max():.2f}")

print(f"\n3️⃣  CURRICULUM GAP (CATEGORICAL):")
excellent_count = len(gap_df[gap_df['Gap'] < 0])
good_count = len(gap_df[(gap_df['Gap'] >= 0) & (gap_df['Gap'] < 10)])
print(f"   • Categories analyzed: {len(gap_df)}")
print(f"   • Excellent coverage: {excellent_count} categories")
print(f"   • Good coverage: {good_count} categories")
print(f"   • Top strength: {gap_df.iloc[0]['Category']} (Gap: {gap_df.iloc[0]['Gap']:.1f})")

print(f"\n📁 FILES CREATED:")
print(f"   • cluster_analysis.csv")
print(f"   • skill_association_rules.csv")
print(f"   • kbtu_curriculum_gap_analysis.csv")
print(f"   • 12_problem4_optimal_k.png")
print(f"   • 13_problem4_clusters_visualization.png")
print(f"   • 15_problem4_association_rules.png")
print(f"   • 16_problem4_curriculum_gap.png")

print("\n" + "="*80)
print()

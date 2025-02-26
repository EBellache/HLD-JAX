import pandas as pd
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, chi2_contingency, ttest_ind
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from core.wavelet_transform import fibonacci_wavelet_transform
from core.sparsification import bistable_sparsification
from core.free_energy import compute_phase_entropy

# Load dataset
file_path = "rstb20190765_si_001.xlsx"
df = pd.read_excel(file_path)

# Extract relevant columns
xfact = df["xfact"]  # Experimental condition (worm type)
data_values = df.drop(columns=["xfact"]).values  # Bioelectric data

# Convert categorical xfact to numerical labels
label_encoder = LabelEncoder()
xfact_labels = label_encoder.fit_transform(xfact)  # 0 = Control, 1 = 1-Week, 2 = 3-Weeks

# Step 1: Apply Fibonacci Wavelet Transform
wavelet_transformed = jnp.array([
    fibonacci_wavelet_transform(sample, jnp.fft.fft, num_scales=10) for sample in data_values
])

# Step 2: Compute Entropy of Wavelet-Transformed Data Before Treatment
entropy_values_before = jnp.array([compute_phase_entropy(sample) for sample in wavelet_transformed])

# Step 3: Apply Bistable Sparsification Treatment to 1-Week and 3-Week Worms
sparsified_data = wavelet_transformed.copy()
sparsified_data[xfact_labels == 1] = bistable_sparsification(sparsified_data[xfact_labels == 1])
sparsified_data[xfact_labels == 2] = bistable_sparsification(sparsified_data[xfact_labels == 2])

# Step 4: Compute Post-Treatment Entropy
entropy_values_after = jnp.array([compute_phase_entropy(sample) for sample in sparsified_data])

# Step 5: Clustering After Treatment
kmeans = KMeans(n_clusters=3, random_state=42)
clusters_treated = kmeans.fit_predict(jnp.asarray(entropy_values_after).reshape(-1, 1))

# Convert results to DataFrame for analysis
results_df = pd.DataFrame({
    "Sample": range(len(entropy_values_after)),
    "Entropy_Before": entropy_values_before,
    "Entropy_After": entropy_values_after,
    "Cluster_Before": clusters_treated,  # Clustering before treatment
    "xfact": xfact,
    "xfact_label": xfact_labels  # Numeric representation of worm type
})

# Step 6: ANOVA Test to Evaluate Treatment Effect
anova_result = f_oneway(
    results_df[results_df["xfact_label"] == 0]["Entropy_After"],  # Control
    results_df[results_df["xfact_label"] == 1]["Entropy_After"],  # 1-Week Treated
    results_df[results_df["xfact_label"] == 2]["Entropy_After"]   # 3-Weeks Treated
)
print("ANOVA Test Results After Treatment:")
print(f"F-statistic: {anova_result.statistic}, p-value: {anova_result.pvalue}")

# Step 7: T-Test to Compare Treated Groups to Control
ttest_1week = ttest_ind(
    results_df[results_df["xfact_label"] == 0]["Entropy_After"],  # Control
    results_df[results_df["xfact_label"] == 1]["Entropy_After"],  # 1-Week Treated
    equal_var=False
)
ttest_3week = ttest_ind(
    results_df[results_df["xfact_label"] == 0]["Entropy_After"],  # Control
    results_df[results_df["xfact_label"] == 2]["Entropy_After"],  # 3-Weeks Treated
    equal_var=False
)

print("\nT-Test Results: Comparison of Treated Worms vs. Control")
print(f"1-Week Treated vs. Control: T-statistic: {ttest_1week.statistic}, p-value: {ttest_1week.pvalue}")
print(f"3-Week Treated vs. Control: T-statistic: {ttest_3week.statistic}, p-value: {ttest_3week.pvalue}")

# Step 8: Chi-Squared Test for Cluster Distribution Change
contingency_table_before = pd.crosstab(results_df["xfact"], results_df["Cluster_Before"])
chi2_stat_before, p_val_before, dof_before, expected_before = chi2_contingency(contingency_table_before)
print("\nChi-Squared Test Results Before Treatment:")
print(f"Chi-Squared Statistic: {chi2_stat_before}, p-value: {p_val_before}")

contingency_table_after = pd.crosstab(results_df["xfact"], results_df["Cluster_Before"])
chi2_stat_after, p_val_after, dof_after, expected_after = chi2_contingency(contingency_table_after)
print("\nChi-Squared Test Results After Treatment:")
print(f"Chi-Squared Statistic: {chi2_stat_after}, p-value: {p_val_after}")

# Step 9: Visualization of Treatment Effects
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(x=results_df["xfact"], y=results_df["Entropy_After"], palette="coolwarm")
plt.xlabel("Worm Type (xfact)")
plt.ylabel("Wavelet-Based Entropy After Treatment")
plt.title("Entropy Differences Post-Treatment")
plt.show()

# Heatmap Visualization Post-Treatment
plt.figure(figsize=(12, 6))
sns.heatmap(jnp.abs(sparsified_data).mean(axis=0), cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap of Wavelet Transform Coefficients After Bistable Sparsification")
plt.xlabel("Wavelet Scale")
plt.ylabel("Sample Index")
plt.show()

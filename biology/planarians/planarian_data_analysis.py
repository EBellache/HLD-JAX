import pandas as pd
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, chi2_contingency
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from core.wavelet_transform import hyperbolic_wavelet_transform
from core.free_energy import compute_phase_entropy

# Load dataset
file_path = "rstb20190765_si_001.xlsx"
df = pd.read_excel(file_path)

# Extract relevant columns
xfact = df["xfact"]  # Experimental condition (worm type)
data_values = df.drop(columns=["xfact"]).values  # Bioelectric data

# Convert categorical xfact to numerical labels
label_encoder = LabelEncoder()
xfact_labels = label_encoder.fit_transform(xfact)  # Assigns 0, 1, 2 for Control, 1-Week, 3-Weeks

# Step 1: Apply Fibonacci Wavelet Transform
wavelet_transformed = jnp.array([
    hyperbolic_wavelet_transform(sample, jnp.fft.fft, num_scales=10) for sample in data_values
])

# Step 2: Compute Entropy of Wavelet-Transformed Data
entropy_values = jnp.array([compute_phase_entropy(sample) for sample in wavelet_transformed])

# Step 3: Clustering to Detect Three Distinct States
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(jnp.asarray(entropy_values).reshape(-1, 1))

# Convert results to DataFrame for analysis
results_df = pd.DataFrame({
    "Sample": range(len(entropy_values)),
    "Entropy": entropy_values,
    "Cluster": clusters,
    "xfact": xfact,
    "xfact_label": xfact_labels  # Numeric representation of worm type
})

# Step 4: ANOVA Test for Statistical Difference in Entropy
anova_result = f_oneway(
    results_df[results_df["xfact_label"] == 0]["Entropy"],
    results_df[results_df["xfact_label"] == 1]["Entropy"],
    results_df[results_df["xfact_label"] == 2]["Entropy"]
)

print("ANOVA Test Results:")
print(f"F-statistic: {anova_result.statistic}, p-value: {anova_result.pvalue}")

# Step 5: Chi-Squared Test for Clustered State Distributions
contingency_table = pd.crosstab(results_df["xfact"], results_df["Cluster"])
chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)

print("\nChi-Squared Test Results:")
print(f"Chi-Squared Statistic: {chi2_stat}, p-value: {p_val}")

# Step 6: Visualization of Entropy Differences
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(x=results_df["xfact"], y=results_df["Entropy"], palette="coolwarm")
plt.xlabel("Worm Type (xfact)")
plt.ylabel("Wavelet-Based Entropy")
plt.title("Entropy Differences Between Experimental Conditions")
plt.show()

# Step 7: Heatmap of Clustered Regenerative States
plt.figure(figsize=(12, 6))
sns.heatmap(jnp.abs(wavelet_transformed).mean(axis=0), cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap of Wavelet Transform Coefficients (Clustered by Worm Type)")
plt.xlabel("Wavelet Scale")
plt.ylabel("Sample Index")
plt.show()

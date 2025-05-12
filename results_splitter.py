import os
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt

def load_scores(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return (
        np.array(data["map"]),
        np.array(data["ndcg"]),
        np.array(data["precision"]),
        np.array(data["recall"]),
        np.array(data["fscore"])
    )

def compare_metrics(row, result_dir="output/results"):
    baseline, variant = row["Comparison"].split(" vs ")
    base_file = os.path.join(result_dir, f"{baseline}_scores_k10.json")
    var_file = os.path.join(result_dir, f"{variant}_scores_k10.json")

    try:
        map_A, ndcg_A, prec_A, recall_A, fscore_A = load_scores(base_file)
        map_B, ndcg_B, prec_B, recall_B, fscore_B = load_scores(var_file)
    except:
        print(f"⚠️ Error loading files for: {row['Comparison']}")
        return pd.Series([False, 0, 0, 0, 0, 0])

    map_diff = np.mean(map_B) - np.mean(map_A)
    ndcg_diff = np.mean(ndcg_B) - np.mean(ndcg_A)
    prec_diff = np.mean(prec_B) - np.mean(prec_A)
    recall_diff = np.mean(recall_B) - np.mean(recall_A)
    fscore_diff = np.mean(fscore_B) - np.mean(fscore_A)

    improved = any(diff > 0 for diff in [map_diff, ndcg_diff, prec_diff, recall_diff, fscore_diff])
    return pd.Series([improved, map_diff, ndcg_diff, prec_diff, recall_diff, fscore_diff])

# Load and apply to summary
df = pd.read_csv("output/hypothesis_results/summary.csv")
df_compare = df.copy()
df_compare[
    ["Improved", "MAP_diff", "nDCG_diff", "Precision_diff", "Recall_diff", "FScore_diff"]
] = df_compare.apply(compare_metrics, axis=1)

# Split datasets
improved_df = df_compare[df_compare["Improved"]]
worsened_df = df_compare[~df_compare["Improved"]]

# Save CSVs
improved_df.to_csv("output/improved_configurations.csv", index=False)
worsened_df.to_csv("output/worsened_configurations.csv", index=False)

# ---- Plotting ----

# Improvement plot
melted_imp = improved_df.melt(
    id_vars="Comparison",
    value_vars=["MAP_diff", "nDCG_diff", "Precision_diff", "Recall_diff", "FScore_diff"],
    var_name="Metric",
    value_name="Improvement"
)

plt.figure(figsize=(14, 6))
sns.barplot(data=melted_imp, x="Comparison", y="Improvement", hue="Metric")
plt.axhline(0, ls='--', color='red')
plt.xticks(rotation=45, ha='right')
plt.title("Improvements over Baseline (Positive Metric Differences)")
plt.tight_layout()
plt.grid(True)
plt.savefig("output/improvement_plot.png")
plt.show()

# Degradation plot
melted_deg = worsened_df.melt(
    id_vars="Comparison",
    value_vars=["MAP_diff", "nDCG_diff", "Precision_diff", "Recall_diff", "FScore_diff"],
    var_name="Metric",
    value_name="Decline"
)

plt.figure(figsize=(14, 6))
sns.barplot(data=melted_deg, x="Comparison", y="Decline", hue="Metric")
plt.axhline(0, ls='--', color='red')
plt.xticks(rotation=45, ha='right')
plt.title("Degradations from Baseline (Negative Metric Differences)")
plt.tight_layout()
plt.grid(True)
plt.savefig("output/degradation_plot.png")
plt.show()

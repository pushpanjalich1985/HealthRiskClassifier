# ─────────────────────────────────────────────
#  visualize.py
#  Generates all 5 plots and saves them to
#  the output/ folder as PNG files
# ─────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix

# ── Global style settings ─────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"]  = 150
plt.rcParams["font.family"] = "DejaVu Sans"


def ensure_output_dir(path="output"):
    """Create the output folder if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 1 — EDA Distributions
# ─────────────────────────────────────────────────────────────────────────────
def plot_eda_distributions(df, save_path="output/eda_distributions.png"):
    """
    Plot histograms for all 8 numeric features.
    Each histogram shows how values are distributed across patients.
    """

    features = ["Age", "BMI", "BloodPressure", "Cholesterol",
                "BloodSugar", "Smoking", "PhysicalActivity", "SleepHours"]

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 8))
    fig.suptitle("Feature Distributions — Health Risk Dataset",
                 fontsize=16, fontweight="bold", y=1.02)

    for ax, feature in zip(axes.flatten(), features):

        sns.histplot(
            data=df,
            x=feature,
            bins=30,
            kde=True,
            ax=ax,
            color="steelblue",
            edgecolor="white"
        )

        mean_val   = df[feature].mean()
        median_val = df[feature].median()

        ax.axvline(mean_val,   color="red",    linestyle="--",
                   linewidth=1.5, label=f"Mean: {mean_val:.1f}")
        ax.axvline(median_val, color="orange", linestyle="-.",
                   linewidth=1.5, label=f"Median: {median_val:.1f}")

        ax.set_title(feature, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 2 — Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df, save_path="output/correlation_heatmap.png"):
    """
    Plot a heatmap of Pearson correlations between all numeric features.
    """

    fig, ax = plt.subplots(figsize=(10, 8))

    corr_matrix = df.corr()

    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 9}
    )

    ax.set_title("Feature Correlation Heatmap",
                 fontsize=14, fontweight="bold", pad=20)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 3 — Confusion Matrix Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_test_pred,
                          save_path="output/confusion_matrix.png"):
    """
    Plot the confusion matrix as a colour-coded heatmap.

    Parameters:
        y_test      : Actual labels from the test set
        y_test_pred : Predicted labels from the model
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    cm = confusion_matrix(y_test, y_test_pred)

    class_names = ["High", "Low", "Medium"]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"}
    )

    ax.set_title("Confusion Matrix — Test Set",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("Actual Label",    fontsize=12)

    accuracy = cm.diagonal().sum() / cm.sum()
    ax.text(0.5, -0.12,
            f"Overall Accuracy: {accuracy:.1%}",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=12, color="darkblue", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 4 — Feature Importance Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names,
                            save_path="output/feature_importance.png"):
    """
    Plot a horizontal bar chart of feature importances from the trained model.

    Parameters:
        model         : Trained DecisionTreeClassifier
        feature_names : List of column names from X_train
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    colors = [
        "#e74c3c" if imp >= sorted(importances)[-3]
        else "#3498db"
        for imp in importance_df["Importance"]
    ]

    bars = ax.barh(
        importance_df["Feature"],
        importance_df["Importance"],
        color=colors,
        edgecolor="white",
        height=0.6
    )

    for bar, val in zip(bars, importance_df["Importance"]):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", fontsize=10
        )

    top3_patch = mpatches.Patch(color="#e74c3c", label="Top 3 features")
    rest_patch = mpatches.Patch(color="#3498db", label="Other features")
    ax.legend(handles=[top3_patch, rest_patch], loc="lower right")

    ax.set_title("Feature Importances — Decision Tree",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score (sum = 1.0)", fontsize=11)
    ax.set_xlim(0, max(importances) + 0.08)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 5 — Decision Tree Diagram  (Fix 2 — large canvas, full tree)
# ─────────────────────────────────────────────────────────────────────────────
def plot_decision_tree(model, feature_names,
                       save_path="output/tree_visualization.png"):
    """
    Render the full trained Decision Tree as a visual diagram.
    Large canvas (120×40 inches at 300 DPI) so no nodes overlap.

    Parameters:
        model         : Trained DecisionTreeClassifier
        feature_names : List of column names from X_train
    """

    # ── Large canvas ──────────────────────────────────────────────────────────
    # 120 inches wide × 40 inches tall at 300 DPI
    # = roughly 36000 × 12000 pixels — every node has room to breathe
    fig, ax = plt.subplots(figsize=(120, 40))

    # Fine-tune margins so nodes don't clip at the edges
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)

    # ── Draw the full tree ────────────────────────────────────────────────────
    plot_tree(
        model,
        feature_names=feature_names,      # show column names not "X[0]"
        class_names=["High", "Low", "Med"],# show class names not "0,1,2"
        filled=True,                       # colour nodes by majority class
        rounded=True,                      # rounded corners
        fontsize=7,                        # slightly smaller text fits better
        ax=ax,
        impurity=True,                     # show Gini score in each node
        proportion=False,                  # show raw counts not proportions
        precision=3                        # decimal places for thresholds
        # NOTE: no max_depth here — we show the full tree
    )

    ax.set_title(
        "Decision Tree — Health Risk Classifier\n"
        "(Blue=High Risk  |  Orange=Low Risk  |  Green=Medium Risk)",
        fontsize=20, fontweight="bold", pad=30
    )

    # ── Save at high resolution ───────────────────────────────────────────────
    # dpi=300 → professional print quality
    # bbox_inches="tight" → no whitespace cropped accidentally
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


# ── Run directly for testing ──────────────────────────────────────────────────
if __name__ == "__main__":
    import joblib
    from preprocess import load_data, encode_labels, split_data
    from train import find_best_depth, train_model, evaluate_model

    # ── Load and prepare data ─────────────────────────────────────────────────
    df                               = load_data()
    df_encoded, risk_map             = encode_labels(df.copy())
    X_train, X_test, y_train, y_test = split_data(df_encoded)

    # ── Train model ───────────────────────────────────────────────────────────
    best_depth  = find_best_depth(X_train, y_train)
    model       = train_model(X_train, y_train, best_depth)
    y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)

    # ── Generate all 5 plots ──────────────────────────────────────────────────
    ensure_output_dir()

    plot_eda_distributions(df)
    plot_correlation_heatmap(df_encoded)
    plot_confusion_matrix(y_test, y_test_pred)
    plot_feature_importance(model, X_train.columns.tolist())
    plot_decision_tree(model, X_train.columns.tolist())

    print("\nAll 5 plots saved to output/")
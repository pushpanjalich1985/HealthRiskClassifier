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
# These apply to every plot in this file
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"]    = 150      # resolution of saved images
plt.rcParams["font.family"]   = "DejaVu Sans"


def ensure_output_dir(path="output"):
    """Create the output folder if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 1 — EDA Distributions
#  Shows the distribution of every feature as a histogram
# ─────────────────────────────────────────────────────────────────────────────
def plot_eda_distributions(df, save_path="output/eda_distributions.png"):
    """
    Plot histograms for all 8 numeric features.
    Each histogram shows how values are distributed across patients.
    """

    # The 8 feature columns — everything except RiskLabel
    features = ["Age", "BMI", "BloodPressure", "Cholesterol",
                "BloodSugar", "Smoking", "PhysicalActivity", "SleepHours"]

    # ── Create a 2×4 grid of subplots ────────────────────────────────────────
    # figsize=(width, height) in inches
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 8))

    # fig.suptitle adds one big title above all subplots
    fig.suptitle("Feature Distributions — Health Risk Dataset",
                 fontsize=16, fontweight="bold", y=1.02)

    # axes.flatten() converts the 2×4 grid into a flat list of 8 axes
    # so we can loop through them easily with zip()
    for ax, feature in zip(axes.flatten(), features):

        # ── Draw the histogram ────────────────────────────────────────────
        # bins=30    → divide the range into 30 bars
        # kde=True   → overlay a smooth density curve (Kernel Density Estimate)
        # color      → bar colour
        # edgecolor  → outline colour of each bar
        sns.histplot(
            data=df,
            x=feature,
            bins=30,
            kde=True,
            ax=ax,
            color="steelblue",
            edgecolor="white"
        )

        # ── Add mean and median lines ─────────────────────────────────────
        mean_val   = df[feature].mean()
        median_val = df[feature].median()

        # axvline draws a vertical line at a specific x value
        ax.axvline(mean_val,   color="red",    linestyle="--",
                   linewidth=1.5, label=f"Mean: {mean_val:.1f}")
        ax.axvline(median_val, color="orange", linestyle="-.",
                   linewidth=1.5, label=f"Median: {median_val:.1f}")

        ax.set_title(feature, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=8)

    # tight_layout automatically adjusts spacing between subplots
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")

    # Always close the figure after saving — frees memory
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 2 — Correlation Heatmap
#  Shows how strongly each pair of features is related
# ─────────────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df, save_path="output/correlation_heatmap.png"):
    """
    Plot a heatmap of Pearson correlations between all numeric features.
    """

    fig, ax = plt.subplots(figsize=(10, 8))

    # ── Calculate correlation matrix ──────────────────────────────────────────
    # .corr() computes Pearson correlation between every pair of columns
    # Result is a 9×9 matrix (including RiskLabel)
    corr_matrix = df.corr()

    # ── Create a mask for the upper triangle ─────────────────────────────────
    # A correlation matrix is symmetric — top-right mirrors bottom-left
    # We mask the upper triangle to avoid showing duplicate information
    mask = np.zeros_like(corr_matrix, dtype=bool)

    # triu_indices returns indices of the upper triangle
    # k=1 means exclude the diagonal itself
    mask[np.triu_indices_from(mask, k=1)] = True

    # ── Draw the heatmap ──────────────────────────────────────────────────────
    sns.heatmap(
        corr_matrix,
        mask=mask,           # hide upper triangle
        annot=True,          # show correlation numbers inside cells
        fmt=".2f",           # format numbers to 2 decimal places
        cmap="coolwarm",     # red=positive, blue=negative correlation
        center=0,            # white = zero correlation
        vmin=-1, vmax=1,     # fix colour scale from -1 to +1
        square=True,         # make cells square shaped
        linewidths=0.5,      # thin lines between cells
        ax=ax,
        annot_kws={"size": 9}
    )

    ax.set_title("Feature Correlation Heatmap",
                 fontsize=14, fontweight="bold", pad=20)

    # Rotate x-axis labels so they don't overlap
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 3 — Confusion Matrix Heatmap
#  Shows how well the model classified each risk level
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

    # ── Compute the confusion matrix ─────────────────────────────────────────
    # Returns a 3×3 matrix
    # cm[i][j] = number of patients whose actual class was i
    #            but model predicted class j
    cm = confusion_matrix(y_test, y_test_pred)

    # Class labels — must match the encoding from preprocess.py
    # 0=High, 1=Low, 2=Medium (alphabetical from our risk_map fix)
    class_names = ["High", "Low", "Medium"]

    # ── Draw heatmap ──────────────────────────────────────────────────────────
    sns.heatmap(
        cm,
        annot=True,           # show numbers inside cells
        fmt="d",              # format as integers (not decimals)
        cmap="Blues",         # light=few, dark=many
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

    # ── Add accuracy annotation ───────────────────────────────────────────────
    # The diagonal sum / total = overall accuracy
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
#  Shows which features the Decision Tree relied on most
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

    # ── Get importances and sort them ────────────────────────────────────────
    importances = model.feature_importances_

    # Create a DataFrame for easy sorting
    importance_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True)
    # ascending=True so the most important appears at the TOP
    # (horizontal bar charts read bottom to top)

    # ── Colour bars by importance level ──────────────────────────────────────
    # Top 3 get a highlighted colour, rest get a muted colour
    colors = [
        "#e74c3c" if imp >= sorted(importances)[-3]   # top 3 → red
        else "#3498db"                                  # rest  → blue
        for imp in importance_df["Importance"]
    ]

    # ── Draw horizontal bar chart ─────────────────────────────────────────────
    bars = ax.barh(
        importance_df["Feature"],
        importance_df["Importance"],
        color=colors,
        edgecolor="white",
        height=0.6
    )

    # ── Add value labels on each bar ─────────────────────────────────────────
    for bar, val in zip(bars, importance_df["Importance"]):
        ax.text(
            bar.get_width() + 0.005,   # x position = end of bar + small gap
            bar.get_y() + bar.get_height() / 2,  # y position = middle of bar
            f"{val:.4f}",
            va="center", fontsize=10
        )

    # ── Legend ────────────────────────────────────────────────────────────────
    top3_patch  = mpatches.Patch(color="#e74c3c", label="Top 3 features")
    rest_patch  = mpatches.Patch(color="#3498db", label="Other features")
    ax.legend(handles=[top3_patch, rest_patch], loc="lower right")

    ax.set_title("Feature Importances — Decision Tree",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score (sum = 1.0)", fontsize=11)
    ax.set_xlim(0, max(importances) + 0.08)

    # Remove top and right border lines (cleaner look)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 5 — Decision Tree Diagram
#  Visual representation of the actual tree structure
# ─────────────────────────────────────────────────────────────────────────────
def plot_decision_tree(model, feature_names,
                       save_path="output/tree_visualization.png"):
    """
    Render the trained Decision Tree as a visual diagram.
    Shows every node, split condition, Gini score, and class prediction.

    Parameters:
        model         : Trained DecisionTreeClassifier
        feature_names : List of column names from X_train
    """

    # Large figure — trees get wide quickly
    fig, ax = plt.subplots(figsize=(28, 12))

    # ── plot_tree is sklearn's built-in tree visualiser ───────────────────────
    plot_tree(
        model,
        feature_names=feature_names,      # show column names instead of "X[0]"
        class_names=["High", "Low", "Med"],# show class names instead of "0,1,2"
        filled=True,                       # colour nodes by majority class
        rounded=True,                      # rounded corners on nodes
        fontsize=9,
        ax=ax,
        impurity=True,                     # show Gini score in each node
        proportion=False,                  # show raw counts not proportions
        precision=3
    )

    ax.set_title(
        "Decision Tree — Health Risk Classifier\n"
        "(Blue=High Risk  |  Orange=Low Risk  |  Green=Medium Risk)",
        fontsize=14, fontweight="bold", pad=20
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


# ── Run directly for testing ──────────────────────────────────────────────────
if __name__ == "__main__":
    import joblib
    from preprocess import load_data, encode_labels, split_data
    from train import find_best_depth, train_model, evaluate_model

    # ── Load and prepare data ────────────────────────────────────────────────
    df                               = load_data()
    df_encoded, risk_map             = encode_labels(df.copy())
    X_train, X_test, y_train, y_test = split_data(df_encoded)

    # ── Train model ──────────────────────────────────────────────────────────
    best_depth  = find_best_depth(X_train, y_train)
    model       = train_model(X_train, y_train, best_depth)
    y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)

    # ── Generate all 5 plots ─────────────────────────────────────────────────
    ensure_output_dir()

    # Pass the original df (before encoding) for EDA plots
    # so labels show "Low/Medium/High" not "0/1/2"
    plot_eda_distributions(df)
    plot_correlation_heatmap(df_encoded)
    plot_confusion_matrix(y_test, y_test_pred)
    plot_feature_importance(model, X_train.columns.tolist())
    plot_decision_tree(model, X_train.columns.tolist())

    print("\nAll 5 plots saved to output/")
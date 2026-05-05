# ─────────────────────────────────────────────
#  main.py
#  Entry point for the Health Risk Classifier
#  Runs the full ML pipeline end to end
#
#  Usage:
#      python main.py
# ─────────────────────────────────────────────

import sys
import os
import time
import joblib

# ── Tell Python to look inside src/ for our modules ──────────────────────────
# __file__ = path of this file (main.py)
# os.path.dirname(__file__) = folder containing main.py (project root)
# os.path.join(..., "src") = full path to the src/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Import all our modules ────────────────────────────────────────────────────
from generate_data import generate_health_data, save_data
from preprocess    import load_data, encode_labels, split_data
from train         import find_best_depth, train_model, evaluate_model, save_model
from visualize     import (ensure_output_dir, plot_eda_distributions,
                           plot_correlation_heatmap, plot_confusion_matrix,
                           plot_feature_importance, plot_decision_tree)


def print_banner():
    """Print a welcome banner when the pipeline starts."""
    print("\n" + "=" * 60)
    print("   HEALTH RISK CLASSIFIER — ML PIPELINE")
    print("   Decision Tree  |  Python  |  Scikit-learn")
    print("=" * 60 + "\n")


def print_phase(number, name):
    """Print a clearly visible phase header."""
    print("\n" + "─" * 60)
    print(f"  PHASE {number} — {name}")
    print("─" * 60)


def main():
    """
    Run the complete ML pipeline from data generation to visualisation.

    Pipeline steps:
        1. Generate synthetic dataset
        2. Load and preprocess data
        3. Train and evaluate Decision Tree
        4. Save trained model
        5. Generate all visualisations
    """

    # Track total runtime
    pipeline_start = time.time()

    print_banner()

    # ── PHASE 1 — Data Generation ─────────────────────────────────────────────
    print_phase(1, "DATA GENERATION")

    phase_start = time.time()

    # Generate 1000 rows of synthetic health data
    df = generate_health_data(n_samples=1000, seed=42)

    # Save to data/health_data.csv
    save_data(df, path="data/health_data.csv")

    print(f"Phase 1 complete ({time.time() - phase_start:.2f}s)")


    # ── PHASE 2 — Preprocessing ───────────────────────────────────────────────
    print_phase(2, "PREPROCESSING")

    phase_start = time.time()

    # Load CSV and run quality checks
    df_loaded = load_data(path="data/health_data.csv")

    # Encode RiskLabel: Low→0, Medium→1, High→2
    # We keep df_loaded intact for EDA plots (text labels look better)
    # df_encoded has numeric labels for model training
    df_encoded, risk_map = encode_labels(df_loaded.copy())

    # Split into train/test sets
    X_train, X_test, y_train, y_test = split_data(df_encoded)

    print(f"Phase 2 complete ({time.time() - phase_start:.2f}s)")


    # ── PHASE 3 — Model Training ──────────────────────────────────────────────
    print_phase(3, "MODEL TRAINING & EVALUATION")

    phase_start = time.time()

    # Find the best tree depth using 5-Fold Cross Validation
    best_depth = find_best_depth(X_train, y_train, depth_range=range(3, 11))

    # Train the final model using the best depth
    model = train_model(X_train, y_train, best_depth)

    # Evaluate on test set — prints accuracy, confusion matrix, report
    y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)

    # Serialise trained model to output/decision_tree.pkl
    save_model(model, path="output/decision_tree.pkl")

    print(f"Phase 3 complete ({time.time() - phase_start:.2f}s)")


    # ── PHASE 4 — Visualisation ───────────────────────────────────────────────
    print_phase(4, "GENERATING VISUALISATIONS")

    phase_start = time.time()

    # Make sure output/ folder exists
    ensure_output_dir("output")

    # Feature names needed for tree diagram and importance chart
    feature_names = X_train.columns.tolist()

    # Plot 1 — EDA histograms (use original df_loaded — text labels)
    print("Generating Plot 1/5 — EDA Distributions...")
    plot_eda_distributions(
        df_loaded,
        save_path="output/eda_distributions.png"
    )

    # Plot 2 — Correlation heatmap (use df_encoded — numeric for .corr())
    print("Generating Plot 2/5 — Correlation Heatmap...")
    plot_correlation_heatmap(
        df_encoded,
        save_path="output/correlation_heatmap.png"
    )

    # Plot 3 — Confusion matrix
    print("Generating Plot 3/5 — Confusion Matrix...")
    plot_confusion_matrix(
        y_test, y_test_pred,
        save_path="output/confusion_matrix.png"
    )

    # Plot 4 — Feature importances
    print("Generating Plot 4/5 — Feature Importance...")
    plot_feature_importance(
        model, feature_names,
        save_path="output/feature_importance.png"
    )

    # Plot 5 — Full decision tree diagram
    print("Generating Plot 5/5 — Decision Tree (large file, please wait)...")
    plot_decision_tree(
        model, feature_names,
        save_path="output/tree_visualization.png"
    )

    print(f"Phase 4 complete ({time.time() - phase_start:.2f}s)")


    # ── PIPELINE COMPLETE ─────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Total runtime : {total_time:.2f} seconds")
    print(f"\n  Output files:")
    print(f"    data/health_data.csv")
    print(f"    output/decision_tree.pkl")
    print(f"    output/eda_distributions.png")
    print(f"    output/correlation_heatmap.png")
    print(f"    output/confusion_matrix.png")
    print(f"    output/feature_importance.png")
    print(f"    output/tree_visualization.png")

    print(f"\n  Risk label mapping used:")
    for label, number in risk_map.items():
        print(f"    {label:8s} → {number}")

    print(f"\n  Model summary:")
    print(f"    Algorithm : Decision Tree Classifier")
    print(f"    Max depth : {best_depth}")
    print(f"    Leaves    : {model.get_n_leaves()}")
    print(f"    Features  : {len(feature_names)}")

    print("\n" + "=" * 60 + "\n")

    return model, risk_map


# ── Entry point guard ─────────────────────────────────────────────────────────
# This ensures main() only runs when you execute this file directly
# If another script imports main.py, main() won't run automatically
if __name__ == "__main__":
    main()
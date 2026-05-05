# ─────────────────────────────────────────────
#  train.py
#  Trains a Decision Tree classifier on the
#  health dataset, tunes max_depth using
#  cross-validation, and evaluates on test set
# ─────────────────────────────────────────────

import os # For creating output folder if it doesn't exist
import joblib # For saving/loading the trained model
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Import our own preprocessing functions from Phase 2
from preprocess import load_data, encode_labels, split_data


def find_best_depth(X_train, y_train, depth_range=range(3, 11)):
    """
    Test multiple tree depths using 5-Fold Cross Validation.
    Returns the depth that gives the highest average CV accuracy.

    Parameters:
        X_train    : Training features
        y_train    : Training labels
        depth_range: Which depths to test (default: 3 to 10)

    Returns:
        best_depth : The depth with highest CV accuracy
    """

    print("=" * 50)
    print("FINDING BEST TREE DEPTH (Cross Validation)")
    print("=" * 50)
    print(f"{'Depth':<10} {'CV Accuracy':<15} {'Std Dev'}")
    print("-" * 40)

    best_depth    = None
    best_score    = 0

    for depth in depth_range:

        # ── Build a tree at this depth ────────────────────────────────────
        # criterion="gini" → use Gini Impurity to find best splits
        # random_state=42  → reproducibility
        model = DecisionTreeClassifier(
            max_depth=depth,
            criterion="gini",
            random_state=42
        )

        # ── 5-Fold Cross Validation ───────────────────────────────────────
        # cv=5       → split training data into 5 folds
        # scoring    → what metric to measure
        # Returns an array of 5 accuracy scores (one per fold)
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=5,
            scoring="accuracy"
        )

        # Average of the 5 fold scores = overall CV accuracy
        mean_score = cv_scores.mean()
        std_score  = cv_scores.std()

        print(f"{depth:<10} {mean_score:.4f}{'':>5} ±{std_score:.4f}")

        # Track the best depth seen so far
        if mean_score > best_score:
            best_score = mean_score
            best_depth = depth

    print("-" * 40)
    print(f"Best depth : {best_depth}  (CV accuracy: {best_score:.4f})")

    return best_depth


def train_model(X_train, y_train, best_depth):
    """
    Train the final Decision Tree using the best depth found.

    Parameters:
        X_train    : Training features
        y_train    : Training labels
        best_depth : The max_depth to use

    Returns:
        model : The trained DecisionTreeClassifier
    """

    print("\n" + "=" * 50)
    print(f"TRAINING FINAL MODEL  (max_depth={best_depth})")
    print("=" * 50)

    model = DecisionTreeClassifier(
        max_depth=best_depth,
        criterion="gini",
        random_state=42
    )

    # ── fit() is where learning actually happens ──────────────────────────
    # The tree analyses X_train and y_train together
    # It figures out the best questions (splits) to separate the classes
    # After this line, the model has learned its decision rules
    model.fit(X_train, y_train)

    print(f"Model trained successfully.")
    print(f"Number of leaves : {model.get_n_leaves()}")
    print(f"Tree depth       : {model.get_depth()}")

    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate the trained model on both training and test sets.
    Prints accuracy, confusion matrix, and classification report.

    Parameters:
        model   : Trained DecisionTreeClassifier
        X_train, X_test, y_train, y_test : Data splits
    """

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    # ── predict() applies the learned rules to new data ───────────────────
    # The model walks each row through its decision tree
    # and returns the predicted class label
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    # ── Accuracy ──────────────────────────────────────────────────────────
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc  = accuracy_score(y_test,  y_test_pred)

    print(f"\nTraining Accuracy : {train_acc:.4f} ({train_acc*100:.1f}%)")
    print(f"Test Accuracy     : {test_acc:.4f}  ({test_acc*100:.1f}%)")

    # ── Overfitting check ─────────────────────────────────────────────────
    gap = train_acc - test_acc
    print(f"Train/Test Gap    : {gap:.4f}")

    if gap < 0.05:
        print("Verdict           : Good fit — no significant overfitting")
    elif gap < 0.10:
        print("Verdict           : Slight overfitting — acceptable")
    else:
        print("Verdict           : Overfitting detected — consider reducing depth")

    # ── Confusion Matrix ──────────────────────────────────────────────────
    # Labels 0=High, 1=Low, 2=Medium (alphabetical from encoding)
    cm = confusion_matrix(y_test, y_test_pred)

    print(f"\nConfusion Matrix (Test Set):")
    print(f"{'':15} Predicted")
    print(f"{'':15} High   Low   Medium")
    labels = ["High", "Low", "Medium"]
    for i, row_label in enumerate(labels):
        row = "  ".join(f"{cm[i][j]:5}" for j in range(len(labels)))
        print(f"Actual {row_label:<9} {row}")

    # ── Classification Report ─────────────────────────────────────────────
    # Shows precision, recall, f1-score per class
    # target_names maps numbers back to text labels
    print(f"\nClassification Report:")
    print(classification_report(
        y_test, y_test_pred,
        target_names=["High", "Low", "Medium"]
    ))

    # ── Feature Importance ────────────────────────────────────────────────
    # After training, the tree knows which features were most useful
    # feature_importances_ gives a score 0-1 for each feature
    # All scores sum to 1.0
    feature_names = X_train.columns.tolist()
    importances   = model.feature_importances_

    print("Feature Importances (how useful each feature was):")
    print("-" * 40)

    # Zip pairs each feature name with its importance score
    # sorted() sorts them from most to least important
    importance_pairs = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],    # sort by the importance value
        reverse=True            # highest first
    )

    for feature, importance in importance_pairs:
        # Visual bar made of █ characters — length = importance × 30
        bar = "█" * int(importance * 30)
        print(f"  {feature:<20} {importance:.4f}  {bar}")

    return y_test_pred


def save_model(model, path="output/decision_tree.pkl"):
    """
    Serialise the trained model to disk using joblib.
    This lets us load and use the model later without retraining.
    """

    # Create the output folder if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # joblib.dump saves the model object to a .pkl file
    # pkl = pickle format = Python's way of saving any object to disk
    joblib.dump(model, path)
    print(f"\nModel saved to {path}")
    print("To load it later: model = joblib.load('output/decision_tree.pkl')")


# ── Run directly for testing ──────────────────────────────────────────────────
if __name__ == "__main__":

    # Step 1 — Load and preprocess data
    df                               = load_data()
    df, risk_map                     = encode_labels(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 2 — Find best depth
    best_depth = find_best_depth(X_train, y_train)

    # Step 3 — Train final model
    model = train_model(X_train, y_train, best_depth)

    # Step 4 — Evaluate
    y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)

    # Step 5 — Save
    save_model(model)   
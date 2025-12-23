import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

import preprocess
import utils
from logistic_regression_model import create_logistic_regression, param_grid as lr_param_grid
from neural_network import create_fnn, param_grid as fnn_param_grid


RANDOM_STATE = 42
PLOTS_DIR = "plots"


def load_for_training(resplit_if_one_class: bool = True):
    """
    Load raw train/test from UNSW files.
    If train has only 1 class and resplit_if_one_class=True, combine and stratified split.
    """
    train_df, test_df = preprocess.load_raw_data()

    if "label" not in train_df.columns or "label" not in test_df.columns:
        raise ValueError("Không tìm thấy cột `label` trong dữ liệu UNSW_NB15.")

    y_train = train_df["label"].astype(int)
    y_test = test_df["label"].astype(int)

    drop_cols = ["label"]
    if "attack_cat" in train_df.columns:
        drop_cols.append("attack_cat")

    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    if resplit_if_one_class and y_train.nunique() < 2:
        print(
            "\n[WARN] File train chỉ có 1 class -> không train được. "
            "Sẽ gộp train+test và chia lại (stratified)."
        )
        X_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)
        y_all = pd.concat([y_train, y_test], axis=0, ignore_index=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            y_all,
            test_size=0.3,
            random_state=RANDOM_STATE,
            stratify=y_all if y_all.nunique() > 1 else None,
        )

    print(f"Train shape: {X_train.shape} | label dist: {np.bincount(y_train)}")
    print(f"Test  shape: {X_test.shape}  | label dist: {np.bincount(y_test)}")
    return X_train, y_train, X_test, y_test


def build_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def tune_and_eval(model_name: str, pipe: Pipeline, param_grid: dict, X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="recall",  # ưu tiên bắt được attack
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)
    print("\nBest params:", gs.best_params_)
    best_model = gs.best_estimator_

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    metrics = utils.evaluate_model(best_model, X_test, y_test, safe_name, out_dir=PLOTS_DIR)

    joblib.dump(best_model, f"{safe_name}_best.joblib")
    print(f"Saved: {safe_name}_best.joblib")
    return metrics, gs.best_params_


def main():
    utils.ensure_dir(PLOTS_DIR)

    X_train, y_train, X_test, y_test = load_for_training(resplit_if_one_class=True)

    # Preprocessor (missing value + encoding + scaling) is part of the pipeline
    preprocessor = preprocess.make_preprocessor(X_train)

    # Optional: some basic plots on raw data (numeric correlation)
    utils.plot_correlation_heatmap(pd.concat([X_train, y_train.rename("label")], axis=1), out_dir=PLOTS_DIR)

    results = []

    # 1) Linear Logistic Regression
    lr = create_logistic_regression(random_state=RANDOM_STATE)
    lr_pipe = build_pipeline(preprocessor, lr)
    lr_metrics, lr_best = tune_and_eval(
        "Linear Logistic Regression",
        lr_pipe,
        lr_param_grid(),
        X_train,
        y_train,
        X_test,
        y_test,
    )
    results.append({"Model": "Linear Logistic Regression", **lr_metrics, "best_params": lr_best})

    # 2) FNN
    fnn = create_fnn(random_state=RANDOM_STATE)
    fnn_pipe = build_pipeline(preprocessor, fnn)
    fnn_metrics, fnn_best = tune_and_eval(
        "FNN (MLPClassifier)",
        fnn_pipe,
        fnn_param_grid(),
        X_train,
        y_train,
        X_test,
        y_test,
    )
    results.append({"Model": "FNN (MLPClassifier)", **fnn_metrics, "best_params": fnn_best})

    summary = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False))
    summary.to_csv("model_results_summary.csv", index=False)
    print("Saved: model_results_summary.csv")


if __name__ == "__main__":
    main()



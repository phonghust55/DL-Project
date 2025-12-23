from sklearn.linear_model import LogisticRegression


def create_logistic_regression(random_state: int = 42) -> LogisticRegression:
    # Linear logistic regression
    return LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=random_state,
    )


def param_grid() -> dict:
    # Keep it small/fast; expand later if needed
    return {
        "model__C": [0.01, 0.1, 1.0, 10.0],
    }



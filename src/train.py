import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)
from scipy.stats import randint
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback

if __package__:
    from .prepare_data import (FEI, HISTORY, PROFILES, counts, encode_comparison, encode_inspection_features, enhanced_history,
                               inspection_category_levels,
                               load_workbook, prepare_inspections, sample_cohort, sha256, split_cohort, retrospective_splits)
    from .validate import CATEGORICAL, FEATURES, NUMERIC, TARGET, validate_split
else:
    from prepare_data import (FEI, HISTORY, PROFILES, counts, encode_comparison, encode_inspection_features, enhanced_history,
                               inspection_category_levels,
                              load_workbook, prepare_inspections, sample_cohort, sha256, split_cohort, retrospective_splits)
    from validate import CATEGORICAL, FEATURES, NUMERIC, TARGET, validate_split

def selected_encoder(X_train, selected):
    """Map the compact indicator names to the same sorted train categories."""
    transformers = []
    needed = set(selected)
    for feature in CATEGORICAL:
        columns = sorted(set(X_train[feature]))
        indices = [i for i in range(len(columns)) if f"{feature}__{i}" in needed]
        if indices:
            # Match only selected values, producing zero for all other categories.
            values = [columns[i] for i in indices]
            encoder = OneHotEncoder(categories=[values], handle_unknown="ignore", sparse_output=False)
            transformers.append((feature, encoder, [feature]))
            needed.difference_update(f"{feature}__{i}" for i in indices)
    numeric = [feature for feature in NUMERIC if feature in needed]
    needed.difference_update(numeric)
    if needed or not (transformers or numeric):
        raise ValueError(f"Invalid selected features: {sorted(needed)}")
    if numeric:
        transformers.append(("numeric", "passthrough", numeric))
    return ColumnTransformer(transformers)


def load_split(folder):
    data = pd.read_parquet(Path(folder) / "rows.parquet")
    validate_split(data)
    return data[FEATURES], data[TARGET]

def eligible_recall(actual, labels):
    if precision_score(actual, labels, zero_division=0) <= actual.mean():
        return 0.0
    return recall_score(actual, labels, zero_division=0)


class LogRecall(TrainingCallback):
    def __init__(self, x_val, y_val):
        self.x_val = x_val
        self.y_val = y_val

    def after_iteration(self, model, epoch, evals_log):
        if (epoch + 1) % 5 == 0:
            labels = model.predict(xgb.DMatrix(self.x_val)) >= 0.5
            mlflow.log_metric("val_oai_recall", eligible_recall(
                self.y_val, labels), step=epoch + 1)
        return False

def train_sweep_trial(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--n-estimators", type=int, required=True)
    parser.add_argument("--subsample", type=float, required=True)
    parser.add_argument("--colsample-bytree", type=float, required=True)
    parser.add_argument("--smote-ratio", type=float, required=True)
    parser.add_argument("--selected-features", default="")
    parser.add_argument("--benchmark", type=int, choices=(0, 1), default=0)
    args = parser.parse_args(argv)

    X_train, y_train = load_split(args.train)
    X_val, y_val = load_split(args.validation)
    if args.selected_features:
        encoder = selected_encoder(X_train, args.selected_features.split(","))
    else:
        encoder = ColumnTransformer(
            [("categorical", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
             ("numeric", "passthrough", NUMERIC)],
            sparse_threshold=1.0,
        )
    train_matrix = encoder.fit_transform(X_train)
    val_matrix = encoder.transform(X_val)
    if y_train.mean() / (1 - y_train.mean()) < args.smote_ratio:
        train_matrix, y_train = SMOTE(
            sampling_strategy=args.smote_ratio, random_state=20, k_neighbors=5
        ).fit_resample(train_matrix, y_train)
    model = XGBClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        learning_rate=args.learning_rate, eval_metric="logloss",
        subsample=args.subsample, colsample_bytree=args.colsample_bytree,
        callbacks=[LogRecall(val_matrix, y_val)], random_state=20, n_jobs=2,
    )

    with mlflow.start_run():
        mlflow.log_params({
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "n_estimators": args.n_estimators,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "smote_ratio": args.smote_ratio,
            "selected_features": args.selected_features,
        })
        if args.benchmark:
            baselines = {
                "logistic": LogisticRegression(max_iter=300, random_state=20),
                "tree": DecisionTreeClassifier(random_state=20),
                "forest": RandomForestClassifier(
                    n_estimators=100, random_state=20, n_jobs=2),
            }
            for name, baseline in baselines.items():
                baseline.fit(train_matrix, y_train)
                p = baseline.predict(val_matrix)
                score = baseline.predict_proba(val_matrix)[:, 1]
                mlflow.log_metrics({
                    f"{name}_accuracy": accuracy_score(y_val, p),
                    f"{name}_recall": recall_score(y_val, p, zero_division=0),
                    f"{name}_precision": precision_score(y_val, p, zero_division=0),
                    f"{name}_f1": f1_score(y_val, p, zero_division=0),
                    f"{name}_roc_auc": roc_auc_score(y_val, score),
                })
        model.fit(train_matrix, y_train,
                  eval_set=[(val_matrix, y_val)], verbose=False)
        model.set_params(callbacks=None)
        pipeline = Pipeline([("encode", encoder), ("model", model)])
        labels = pipeline.predict(X_val)
        scores = pipeline.predict_proba(X_val)[:, 1]
        metrics = {
            "val_oai_recall": eligible_recall(y_val, labels),
            "final_val_accuracy": accuracy_score(y_val, labels),
            "final_val_oai_recall": recall_score(y_val, labels, zero_division=0),
            "val_f1": f1_score(y_val, labels, zero_division=0),
            "val_roc_auc": roc_auc_score(y_val, scores),
            "val_average_precision": average_precision_score(y_val, scores),
            "val_oai_precision": precision_score(y_val, labels, zero_division=0),
        }
        mlflow.log_metrics(metrics, step=args.n_estimators)
        diagnostics = Path(args.model) / "diagnostics"
        diagnostics.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"actual_label": y_val, "predicted_label": labels,
                      "predicted_prob": scores, "correct": y_val.to_numpy() == labels}
                     ).to_csv(diagnostics / "validation_predictions.csv", index=False)
        pd.DataFrame({"feature": encoder.get_feature_names_out(),
                      "importance": model.feature_importances_}
                     ).sort_values("importance", ascending=False).to_csv(
                         diagnostics / "model_feature_importances.csv", index=False)
        (diagnostics / "model_summary.txt").write_text(str(model), encoding="utf8")
        (diagnostics / "confusion_matrix.json").write_text(json.dumps(
            confusion_matrix(y_val, labels, labels=[0, 1]).tolist()), encoding="utf8")
        sample = X_train.head(5)
        mlflow.sklearn.save_model(
            pipeline, str(Path(args.model) / "mlflow-model"),
            signature=infer_signature(sample, pipeline.predict(sample)),
            input_example=sample,
            pip_requirements=[
                "mlflow==2.15.1", "setuptools==80.9.0", "pandas==2.2.3",
                "scikit-learn==1.5.2", "xgboost==2.1.4",
            ],
        )

RATIOS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
DECK = {
    "XGBoost": {"accuracy": .74, "roc_auc": .83, "precision": .12, "recall": .78},
    "Random Forest": {"accuracy": .74, "roc_auc": .81, "precision": .12, "recall": .75},
    "Logistic Regression": {"accuracy": .75, "roc_auc": .80, "precision": .11, "recall": .75},
    "Decision Tree": {"accuracy": .85, "roc_auc": .82, "precision": .15, "recall": .57},
}


def oversample(X, y, ratio):
    if ratio is None or y.sum() / (len(y) - y.sum()) >= ratio:
        return X, y
    return SMOTE(sampling_strategy=ratio, random_state=20, k_neighbors=5).fit_resample(X, y)


def evaluation_xgb(**params):
    return XGBClassifier(random_state=20, eval_metric="logloss", n_jobs=2, **params)


def metrics(y, labels, scores):
    tn, fp, fn, tp = confusion_matrix(y, labels, labels=[0, 1]).ravel()
    return {
        "recall": recall_score(y, labels, zero_division=0),
        "precision": precision_score(y, labels, zero_division=0),
        "accuracy": accuracy_score(y, labels),
        "f1": f1_score(y, labels, zero_division=0),
        "roc_auc": roc_auc_score(y, scores),
        "average_precision": average_precision_score(y, scores),
        "false_positives": int(fp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def evaluate(model, X, y, threshold=None):
    scores = model.predict_proba(X)[:, 1]
    labels = model.predict(X) if threshold is None else scores >= threshold
    return metrics(y, labels, scores)


def top_sets(X, y):
    rankers = {
        "DT_top": DecisionTreeClassifier(random_state=20),
        "RF_top": RandomForestClassifier(random_state=20, n_jobs=2),
        "XGB_top": evaluation_xgb(),
    }
    subsets = {}
    for name, model in rankers.items():
        model.fit(X, y)
        ranking = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        subsets[name] = ranking.sort_values("importance", ascending=False)["feature"].head(10).tolist()
    sets = list(map(set, subsets.values()))
    subsets["COMMON"] = sorted(set.intersection(*sets))
    subsets["ALL"] = sorted(set.union(*sets))
    return {name: sorted(cols) for name, cols in subsets.items() if cols}


def deck_baseline(X_train, y_train, X_compare, y_compare, *, reproduce_search_auc=False):
    ratio_report = []
    for ratio in RATIOS:
        X_fit, y_fit = oversample(X_train, y_train, ratio)
        model = evaluation_xgb().fit(X_fit, y_fit)
        result = evaluate(model, X_compare, y_compare)
        if reproduce_search_auc:
            result["roc_auc"] = roc_auc_score(y_compare, model.predict(X_compare))
        ratio_report.append({"ratio": ratio, **result})
    ratio = pd.DataFrame(ratio_report).sort_values("recall", ascending=False)["ratio"].iloc[1]
    X_fit, y_fit = oversample(X_train, y_train, ratio)
    subsets = top_sets(X_fit, y_fit)
    features_report = []
    for name, cols in subsets.items():
        subset_fit, y_subset = oversample(X_train[cols], y_train, ratio)
        probe = evaluation_xgb().fit(subset_fit, y_subset)
        features_report.append({"feature_set": name, "columns": cols,
                                **evaluate(probe, X_compare[cols], y_compare)})
    columns = subsets["XGB_top"]
    return columns, float(ratio), ratio_report, features_report


def reproduce_exported_tuning(X_fit, y_fit, X_test, y_test):
    # The original search seeded its sampler but not the three candidate grids.
    # Seed grid generation explicitly so this reproduction is repeatable.
    rng = np.random.RandomState(20)
    search = RandomizedSearchCV(
        evaluation_xgb(),
        param_distributions={
            "n_estimators": randint(10, 100),
            "max_depth": randint(1, 15),
            "learning_rate": rng.uniform(.01, .30, 20),
            "subsample": rng.uniform(.60, 1.00, 20),
            "colsample_bytree": rng.uniform(.60, 1.00, 20),
        },
        cv=5, scoring="recall", n_iter=25, random_state=20, n_jobs=1,
    )
    print("Reproducing the original tuned XGBoost search.", flush=True)
    search.fit(X_fit, y_fit)
    return {
        "best_parameters": {key: value.item() if isinstance(value, np.generic) else value
                            for key, value in search.best_params_.items()},
        "cv_recall": float(search.best_score_),
        "test": evaluate(search.best_estimator_, X_test, y_test),
        "original_export": {"recall": 0.7569230769230769, "precision": 0.11443634672042177,
                            "roc_auc": 0.8178517911977807},
        "parameter_grid_seed": 20,
        "protocol": "25 recall-scored candidates, 5 folds after training SMOTE, historical reproduction only.",
    }


def deck_parity(frame, cohort):
    inputs = pd.get_dummies(frame[["sum_posted_citations", *HISTORY, *PROFILES]],
                            columns=PROFILES, drop_first=True).fillna(0).sort_index(axis=1)
    train, test = split_cohort(cohort, .25, 20)
    X_train, X_test = inputs.loc[train.index], inputs.loc[test.index]
    y_train, y_test = train[TARGET], test[TARGET]
    cols, ratio, ratios, subsets = deck_baseline(
        X_train, y_train, X_test, y_test, reproduce_search_auc=True,
    )
    X_fit, y_fit = oversample(X_train[cols], y_train, ratio)
    models = {
        "Logistic Regression": LogisticRegression(random_state=20, max_iter=500),
        "Decision Tree": DecisionTreeClassifier(random_state=20),
        "Random Forest": RandomForestClassifier(random_state=20, n_jobs=2),
        "XGBoost": evaluation_xgb(),
    }
    comparison = []
    for name, model in models.items():
        model.fit(X_fit, y_fit)
        result = evaluate(model, X_test[cols], y_test)
        comparison.append({
            "model": name, **result, "deck_metrics": DECK[name],
            "delta_from_deck": {key: result[key] - expected for key, expected in DECK[name].items()},
        })
    return {"train": counts(train), "test": counts(test), "resampled_train_rows": len(y_fit),
            "selected_smote_ratio": ratio, "selected_features": cols,
            "smote_search": ratios, "subset_search": subsets, "model_comparison": comparison,
            "tuned_xgboost_reproduction": reproduce_exported_tuning(X_fit, y_fit, X_test[cols], y_test)}


def choose_threshold(y, scores, precision_target):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    eligible = np.flatnonzero((precision[:-1] > precision_target) & (recall[:-1] > 0))
    if len(eligible):
        i = max(eligible, key=lambda k: (recall[k], precision[k]))
        return float(thresholds[i]), True
    return .5, False


class InspectionRiskModel(mlflow.pyfunc.PythonModel):
    def __init__(self, estimator, threshold, numbers, levels, selected_columns):
        self.estimator = estimator
        self.threshold = threshold
        self.numbers = numbers
        self.levels = levels
        self.selected_columns = selected_columns

    def predict(self, context, model_input, params=None):
        encoded = encode_inspection_features(model_input, self.numbers, self.levels)
        scores = self.estimator.predict_proba(encoded[self.selected_columns])[:, 1]
        return (scores >= self.threshold).astype("int64")


def candidate_inputs(frame, numbers, levels):
    features = frame[[*numbers, *levels]].copy()
    features[numbers] = features[numbers].fillna(0).astype("float64")
    for name in levels:
        features[name] = features[name].fillna("").astype(str)
    return features.reset_index(drop=True)


def save_candidate(output, reports, model, selected, numbers, train, holdout, expected):
    levels = inspection_category_levels(train)
    serving = InspectionRiskModel(model, selected["threshold"], numbers, levels, selected["columns"])
    sample = candidate_inputs(train.head(2), numbers, levels)
    predictions = serving.predict(None, sample)
    path = output / "mlflow-model"
    path.parent.mkdir(parents=True, exist_ok=True)
    dependencies = ["mlflow", "pandas", "numpy", "scikit-learn", "xgboost", "cloudpickle", "setuptools"]
    mlflow.pyfunc.save_model(
        str(path), python_model=serving, input_example=sample,
        code_paths=[str(Path(__file__).with_name(name)) for name in ("prepare_data.py", "validate.py")],
        signature=infer_signature(sample, predictions),
        pip_requirements=[f"{name}=={version(name)}" for name in dependencies],
    )
    loaded = mlflow.pyfunc.load_model(str(path))
    replayed = loaded.predict(candidate_inputs(holdout, numbers, levels))
    if not np.array_equal(replayed, expected):
        raise RuntimeError("Packaged candidate predictions differ from the evaluated holdout predictions.")
    (reports / "sample-request.json").write_text(
        json.dumps({"input_data": json.loads(sample.to_json(orient="split"))}), encoding="utf-8"
    )
    (reports / "candidate.json").write_text(
        json.dumps({"sample_predictions": predictions.tolist(), "threshold": selected["threshold"],
                    "inputs": list(sample.columns), "context": "retrospective"}, indent=2),
        encoding="utf-8",
    )


def improve(cohort, trials, precision_target, *, model_output=None, reports=None):
    development, holdout = split_cohort(cohort, .25, 41)
    train, val = split_cohort(development, .2, 20)
    y_train, y_val, y_holdout = train[TARGET], val[TARGET], holdout[TARGET]
    baseline_numbers = ["sum_posted_citations", *HISTORY]
    extra = ["prior_citation_total", "prior_citations_per_insp", "has_prior_inspection",
             "previous_inspection_citations", "previous_oai_recent"]
    matrices = {
        "original": encode_comparison(train, val, holdout, baseline_numbers),
        "expanded": encode_comparison(train, val, holdout, baseline_numbers + extra),
    }
    X_train, X_val, _ = matrices["original"]
    cols, ratio, ratios, baseline_subsets = deck_baseline(X_train, y_train, X_val, y_val)
    X_fit, y_fit = oversample(X_train[cols], y_train, ratio)
    baseline = evaluation_xgb().fit(X_fit, y_fit)
    baseline_validation = evaluate(baseline, X_val[cols], y_val)
    baseline_val_scores = baseline.predict_proba(X_val[cols])[:, 1]
    baseline_threshold, baseline_precision_met = choose_threshold(
        y_val, baseline_val_scores, precision_target,
    )

    # Cover each engineered feature set with the default XGBoost configuration
    # before the randomized weighting, resampling and hyperparameter search.
    candidates = {("original", row["feature_set"]): row["columns"] for row in baseline_subsets}
    candidates[("original", "FULL")] = matrices["original"][0].columns.tolist()
    expanded_train = matrices["expanded"][0]
    ranked_fit, ranked_y = oversample(expanded_train, y_train, ratio)
    for subset, columns in top_sets(ranked_fit, ranked_y).items():
        candidates[("expanded", subset)] = columns
    candidates[("expanded", "FULL")] = expanded_train.columns.tolist()
    settings = [
        {"candidate": key, "smote": ratio, "scale_pos_weight": 1.0}
        for key in candidates
    ]
    search = {
        "candidate": list(candidates), "smote": [None, *RATIOS],
        "scale_pos_weight": [1.0, 2.0, 4.0, 8.0, 16.0],
        "n_estimators": [50, 100, 180], "max_depth": [2, 4, 6, 9, 13],
        "learning_rate": [.025, .05, .1, .2], "subsample": [.7, .85, 1.0],
        "colsample_bytree": [.7, .85, 1.0], "min_child_weight": [1, 3, 8],
    }
    settings.extend(ParameterSampler(search, n_iter=trials, random_state=20))
    report, best = [], None
    for trial_number, params in enumerate(settings, start=1):
        design, subset = params["candidate"]
        columns = candidates[(design, subset)]
        X_train, X_val, _ = matrices[design]
        X_fit, y_fit = oversample(X_train[columns], y_train, params["smote"])
        model_params = {key: value for key, value in params.items() if key not in ("candidate", "smote")}
        model = evaluation_xgb(**model_params).fit(X_fit, y_fit)
        scores = model.predict_proba(X_val[columns])[:, 1]
        threshold, meets_precision = choose_threshold(y_val, scores, precision_target)
        result = metrics(y_val, scores >= threshold, scores)
        row = {"feature_design": design, "feature_set": subset, "columns": columns,
               "parameters": model_params, "smote": params["smote"], "threshold": threshold,
               "meets_validation_precision_target": meets_precision, **result}
        report.append(row)
        print(
            f"Trial {trial_number}/{len(settings)} {design}/{subset} "
            f"validation recall={result['recall']:.4f} precision={result['precision']:.4f}",
            flush=True,
        )
        key = (meets_precision, result["recall"], result["precision"], result["average_precision"])
        if best is None or key > best[0]:
            best = key, row, model

    _, selected, model = best
    baseline_holdout = evaluate(baseline, matrices["original"][2][cols], y_holdout)
    holdout_features = matrices[selected["feature_design"]][2][selected["columns"]]
    holdout_scores = model.predict_proba(holdout_features)[:, 1]
    holdout_labels = holdout_scores >= selected["threshold"]
    enhanced_holdout = metrics(y_holdout, holdout_labels, holdout_scores)
    if model_output is not None:
        numbers = baseline_numbers + extra if selected["feature_design"] == "expanded" else baseline_numbers
        save_candidate(model_output, reports, model, selected, numbers, train, holdout, holdout_labels)
    baseline_threshold_holdout = evaluate(
        baseline, matrices["original"][2][cols], y_holdout, baseline_threshold,
    )
    return {
        "train": counts(train), "validation": counts(val), "holdout": counts(holdout),
        "holdout_fei_sha256": hashlib.sha256(
            json.dumps(sorted(map(str, holdout[FEI].unique()))).encode()
        ).hexdigest(),
        "baseline": {"columns": cols, "smote": ratio, "validation": baseline_validation,
                     "holdout": baseline_holdout},
        "baseline_threshold_only": {
            "threshold": baseline_threshold,
            "meets_validation_precision_target": baseline_precision_met,
            "validation": metrics(y_val, baseline_val_scores >= baseline_threshold, baseline_val_scores),
            "holdout": baseline_threshold_holdout,
        },
        "enhanced": {"validation": selected, "holdout": enhanced_holdout},
        "precision_target": precision_target, "recall_target": .78,
        "meets_holdout_targets": bool(enhanced_holdout["recall"] > .78
                                      and enhanced_holdout["precision"] > precision_target),
        "baseline_smote_search": ratios, "baseline_subset_search": baseline_subsets,
        "enhancement_trials": report,
    }


def run_evaluation(mode, argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspections", type=Path, required=True)
    parser.add_argument("--citations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--precision-target", type=float, default=.12)
    args = parser.parse_args(argv)
    if args.trials < 0 or not 0 < args.precision_target < 1:
        parser.error("trials must be nonnegative and precision target must be between 0 and 1")
    raw, citations = load_workbook(args.inspections), load_workbook(args.citations)
    inspection = prepare_inspections(raw)
    enhanced = enhanced_history(inspection)
    cohort = sample_cohort(enhanced)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "inspection_workbook": {"sha256": sha256(args.inspections), "rows": len(raw)},
        "citation_workbook": {"sha256": sha256(args.citations), "rows": len(citations),
                              "deduplicated_rows": len(citations.drop_duplicates(keep=False))},
        "engineered": counts(inspection), "sampled": counts(cohort),
        "versions": {package: version(package) for package in
                     ("pandas", "scikit-learn", "xgboost", "imbalanced-learn", "numpy")},
        "prediction_context": "Retrospective. Includes same-inspection posted citation count.",
    }
    with mlflow.start_run():
        mlflow.log_params({"evaluation_mode": mode, "trials": args.trials,
                           "precision_target": args.precision_target})
        if mode in ("evaluate", "improve"):
            print("Running validation-only search on the fixed FEI cohort.", flush=True)
            improvement = improve(cohort, args.trials, args.precision_target,
                                  model_output=args.model_output, reports=args.output)
            (args.output / "holdout_comparison.json").write_text(
                json.dumps(improvement, indent=2), encoding="utf-8"
            )
            for name in ("baseline", "baseline_threshold_only", "enhanced"):
                row = improvement[name]["holdout"]
                mlflow.log_metrics({f"{name}_holdout_{key}": row[key]
                                    for key in ("recall", "precision", "false_positives")})
            print({name: improvement[name]["holdout"] for name in ("baseline", "enhanced")})
        if mode in ("evaluate", "reproduce"):
            print("Reproducing the original deck comparison.", flush=True)
            parity = deck_parity(inspection, cohort)
            (args.output / "deck_parity.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")
            print(pd.DataFrame(parity["model_comparison"])[["model", "recall", "precision", "roc_auc"]])
        (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def select_shared_features(X, y, validation, y_val):
    """Evaluate compact local-style subsets, then share one selection across Azure branches."""
    fitted, labels = oversample(X, y, .7)
    subsets = top_sets(fitted, labels)
    rows = []
    for name, columns in subsets.items():
        fit, fit_y = oversample(X[columns], y, .7)
        model = evaluation_xgb().fit(fit, fit_y)
        scores = model.predict_proba(validation[columns])[:, 1]
        threshold, eligible = choose_threshold(y_val, scores, .12)
        result = metrics(y_val, scores >= threshold, scores)
        rows.append({"feature_set": name, "columns": columns, "eligible": eligible, **result})
    best = max(rows, key=lambda r: (r["eligible"], r["recall"], r["precision"], r["roc_auc"]))
    return {**best, "comparison": rows}


def reference_results(inspections):
    frame = prepare_inspections(load_workbook(inspections))
    cohort = sample_cohort(enhanced_history(frame))
    splits = retrospective_splits(cohort)
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    X, V, T = encode_comparison(train, validation, test, ["sum_posted_citations", *HISTORY])
    cols, ratio, _, _ = deck_baseline(X, train[TARGET], V, validation[TARGET])
    fitted, labels = oversample(X[cols], train[TARGET], ratio)
    model = evaluation_xgb().fit(fitted, labels)
    valid_scores = model.predict_proba(V[cols])[:, 1]
    threshold, eligible = choose_threshold(validation[TARGET], valid_scores, .12)
    return {
        "deck": deck_parity(frame, cohort),
        "matched_baseline": evaluate(model, T[cols], test[TARGET]),
        "threshold_only": evaluate(model, T[cols], test[TARGET], threshold),
        "threshold_only_threshold": threshold,
        "threshold_only_met_validation_precision": eligible,
    }


class LogTargetRecall(LogRecall):
    def after_iteration(self, model, epoch, evals_log):
        if (epoch + 1) % 5 == 0:
            scores = model.predict(xgb.DMatrix(self.x_val))
            threshold, eligible = choose_threshold(self.y_val, scores, .12)
            recall = recall_score(self.y_val, scores >= threshold, zero_division=0)
            mlflow.log_metric("val_oai_recall", recall if eligible else 0.0, step=epoch + 1)
        return False


class ThresholdClassifier(mlflow.pyfunc.PythonModel):
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def predict(self, context, model_input, params=None):
        return (self.model.predict_proba(model_input)[:, 1] >= self.threshold).astype("int64")


def train_capstone_trial(argv):
    parser = argparse.ArgumentParser()
    for name in ("train", "validation", "model"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--trees", type=int, required=True)
    for name in ("rate", "smote", "weight", "row-fraction", "column-fraction"):
        parser.add_argument(f"--{name}", type=float, required=True)
    args = parser.parse_args(argv)
    train = pd.read_parquet(args.train / "rows.parquet")
    validation = pd.read_parquet(args.validation / "rows.parquet")
    y, y_val = train.pop(TARGET), validation.pop(TARGET)
    if list(train.columns) != list(validation.columns):
        raise ValueError("Training and validation schemas differ.")
    X, labels = oversample(train, y, args.smote or None)
    params = dict(max_depth=args.depth, n_estimators=args.trees, learning_rate=args.rate,
                  scale_pos_weight=args.weight, subsample=args.row_fraction,
                  colsample_bytree=args.column_fraction)
    with mlflow.start_run():
        mlflow.log_params({**params, "smote_ratio": args.smote})
        model = evaluation_xgb(**params, callbacks=[LogTargetRecall(validation, y_val)])
        model.fit(X, labels, verbose=False)
        model.set_params(callbacks=None)
        scores = model.predict_proba(validation)[:, 1]
        threshold, eligible = choose_threshold(y_val, scores, .12)
        result = metrics(y_val, scores >= threshold, scores)
        mlflow.log_metric("val_oai_recall", result["recall"] if eligible else 0.0, step=args.trees)
        mlflow.log_metrics({f"final_val_{key}": result[key] for key in
                            ("recall", "precision", "roc_auc", "accuracy")})
        mlflow.log_param("threshold", threshold)
        package = ThresholdClassifier(model, threshold)
        example = train.head(2)
        args.model.mkdir(parents=True, exist_ok=True)
        destination = args.model / "mlflow-model"
        mlflow.pyfunc.save_model(
            str(destination), python_model=package, input_example=example,
            signature=infer_signature(example, package.predict(None, example)),
            pip_requirements=[f"{name}=={version(name)}" for name in
                              ("mlflow", "pandas", "numpy", "scikit-learn", "xgboost", "cloudpickle", "setuptools")],
        )
        restored = mlflow.pyfunc.load_model(str(destination))
        if not np.array_equal(restored.predict(validation), scores >= threshold):
            raise RuntimeError("Saved model predictions differ from the validation predictions.")


def main(argv=None):
    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument("--mode", choices=("sweep", "capstone-trial", "evaluate", "reproduce", "improve"), default="sweep")
    options, remaining = mode_parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if options.mode == "sweep":
        train_sweep_trial(remaining)
    elif options.mode == "capstone-trial":
        train_capstone_trial(remaining)
    else:
        run_evaluation(options.mode, remaining)



if __name__ == "__main__":
    main()

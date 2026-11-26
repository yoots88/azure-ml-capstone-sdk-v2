"""Numeric MLTable inputs for the managed AutoML reader."""
import hashlib
import json

import pandas as pd
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.core.exceptions import ResourceNotFoundError

from src.validate import CATEGORICAL, NUMERIC, TARGET, validate_split


def fit_categories(train):
    return {name: sorted(set(train[name])) for name in CATEGORICAL}


def encode_features(frame, categories):
    selected = categories.get("_selected")
    features = frame.loc[:, NUMERIC].astype("float64")
    if selected is not None:
        features = features.loc[:, [name for name in NUMERIC if name in selected]]
    indicators = {
        f"{name}__{index}": frame[name].eq(value).fillna(False).to_numpy(dtype="int8")
        for name in CATEGORICAL
        for index, value in enumerate(categories[name])
        if (index > 0 or not categories.get("_drop_first"))
        and (selected is None or f"{name}__{index}" in selected)
    }
    encoded = pd.concat([features, pd.DataFrame(indicators, index=frame.index)], axis=1)
    if selected is None:
        return encoded
    subset = encoded.loc[:, selected]
    return subset.astype("float64") if categories.get("_drop_first") else subset


def compare_feature_sets(train, validation, categories):
    """Rank on training only and compare feature sets on untouched validation."""
    from imblearn.over_sampling import SMOTE
    from scipy.sparse import csr_matrix
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier

    train_encoded = encode_features(train, categories)
    val_encoded = encode_features(validation, categories)
    y_train, y_val = train[TARGET], validation[TARGET]
    weights = compute_sample_weight("balanced", y_train)
    ranking_values = train_encoded.to_numpy(dtype="float32")
    ranking_matrix = csr_matrix(ranking_values)
    model_params = dict(n_estimators=54, max_depth=4, learning_rate=0.1,
                        tree_method="hist", random_state=20, n_jobs=2)
    rankers = {
        "DT_top": DecisionTreeClassifier(max_depth=12, random_state=20),
        "RF_top": RandomForestClassifier(n_estimators=60, max_depth=12, random_state=20, n_jobs=2),
        "XGB_top": XGBClassifier(**model_params),
    }
    subsets = {}
    for name, ranker in rankers.items():
        ranker.fit(ranking_values if name == "XGB_top" else ranking_matrix,
                   y_train, sample_weight=weights)
        ranked = sorted(zip(train_encoded.columns, ranker.feature_importances_),
                        key=lambda pair: (-pair[1], pair[0]))
        subsets[name] = [column for column, _ in ranked[:10]]
    selected_sets = [set(columns) for columns in subsets.values()]
    subsets["COMMON"] = sorted(set.intersection(*selected_sets))
    subsets["ALL"] = sorted(set.union(*selected_sets))

    results = []
    for name, columns in subsets.items():
        if not columns:
            continue
        # XGBoost treats absent sparse entries as missing. Keep explicit zeros
        # in these compact candidate matrices, as in the Azure training job.
        X_train = train_encoded[columns].to_numpy(dtype="float32")
        X_val = val_encoded[columns].to_numpy(dtype="float32")
        for ratio in (0.2, 0.5, 1.0):
            X_fit, y_fit = X_train, y_train
            if y_train.sum() / (len(y_train) - y_train.sum()) < ratio:
                X_fit, y_fit = SMOTE(sampling_strategy=ratio, random_state=20).fit_resample(X_train, y_train)
            model = XGBClassifier(**model_params).fit(X_fit, y_fit)
            labels = model.predict(X_val)
            scores = model.predict_proba(X_val)[:, 1]
            precision = precision_score(y_val, labels, zero_division=0)
            results.append({
                "feature_set": name, "columns": columns, "feature_count": len(columns),
                "smote_ratio": ratio, "recall": recall_score(y_val, labels, zero_division=0),
                "precision": precision,
                "f1": f1_score(y_val, labels, zero_division=0),
                "accuracy": accuracy_score(y_val, labels),
                "roc_auc": roc_auc_score(y_val, scores),
                "eligible": bool(labels.any() and not labels.all()
                                 and precision > y_val.mean()),
            })
    return sorted(results, key=lambda row: (
        -row["eligible"], -row["recall"], -row["precision"], -row["roc_auc"], row["feature_count"],
    ))


def register_numeric_inputs(client, mltable, get_asset, output_dir, allow_create):
    frames, versions, lineage = {}, {}, {}
    for split in ("train", "validation"):
        source = get_asset(split)
        frames[split] = mltable.load(
            f"azureml:/{source.id}", ml_client=client
        ).to_pandas_dataframe()
        validate_split(frames[split])
        versions[split] = str(source.version)
        lineage[split] = source.id
    categories = fit_categories(frames["train"])
    categories["_drop_first"] = True
    cache = output_dir / "recall_feature_search.json"
    if cache.is_file():
        report = json.loads(cache.read_text(encoding="utf-8"))
        if report["sources"] != lineage or report["format"] != 1:
            raise ValueError("Feature search cache points to a different workflow. Use a new output folder.")
    else:
        if not allow_create:
            raise FileNotFoundError(f"Missing recall feature search: {cache}")
        results = compare_feature_sets(frames["train"], frames["validation"], categories)
        if not results[0]["eligible"]:
            raise RuntimeError("No feature set beat the validation prevalence at a nontrivial operating point.")
        report = {"format": 1, "sources": lineage, "results": results}
        cache.write_text(json.dumps(report, indent=2), encoding="utf-8")
    best = report["results"][0]
    categories["_selected"] = best["columns"]
    categories["_feature_set"] = best["feature_set"]
    categories["_smote_ratio"] = best["smote_ratio"]
    print(pd.DataFrame(report["results"]).drop(columns="columns").to_string(index=False))
    print("Selected predictors:", categories["_selected"])
    fingerprint = hashlib.sha256(json.dumps(
        {"format": 3, "sources": lineage, "categories": categories},
        sort_keys=True,
    ).encode("utf-8")).hexdigest()
    refs = {}
    for split, frame in frames.items():
        name = f"fda-automl-recall-{split}"
        version = versions[split]
        try:
            asset = client.data.get(name=name, version=version)
        except ResourceNotFoundError:
            if not allow_create:
                raise RuntimeError("Enable SUBMIT_AUTO before creating numeric AutoML assets.") from None
            folder = output_dir / "automl_top10" / split
            folder.mkdir(parents=True, exist_ok=True)
            encoded = encode_features(frame, categories)
            labels = frame[TARGET].astype("int64")
            if split == "train" and labels.sum() / (len(labels) - labels.sum()) < best["smote_ratio"]:
                from imblearn.over_sampling import SMOTE

                encoded, labels = SMOTE(
                    sampling_strategy=best["smote_ratio"], random_state=20,
                ).fit_resample(encoded.astype("float64"), labels)
            encoded[TARGET] = labels.to_numpy()
            encoded.to_parquet(folder / "rows.parquet", engine="pyarrow", index=False)
            (folder / "MLTable").write_text(
                "paths:\n  - file: ./rows.parquet\ntransformations:\n  - read_parquet: {}\n",
                encoding="utf-8",
            )
            asset = client.data.create_or_update(Data(
                name=name, version=version, path=str(folder), type=AssetTypes.MLTABLE,
                tags={"fda_encoding_fingerprint": fingerprint},
            ))
        if asset.type != AssetTypes.MLTABLE or (asset.tags or {}).get(
            "fda_encoding_fingerprint"
        ) != fingerprint:
            raise ValueError(f"Conflicting numeric asset {name}:{version}. Check its lineage.")
        refs[split] = f"azureml:{name}:{version}"
    return refs, categories, fingerprint

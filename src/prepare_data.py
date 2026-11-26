import hashlib
import json
from pathlib import Path

from math import ceil

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import resample, shuffle

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/Data_FDA_Inspections_FY_09_26.xlsx"
CITATIONS = ROOT / "data/raw/Data_FDA_Citations_FY_09_26.xlsx"
OUT = ROOT / "data/prepared"
if __package__:
    from .validate import CATEGORICAL, FEATURES, SCHEMA_VERSION, TARGET, validate_split
else:
    from validate import CATEGORICAL, FEATURES, SCHEMA_VERSION, TARGET, validate_split

PROFILE = ["country_area", "state", "project_area", "product_type", "additional_details"]
KEY = ["fei_number", "inspection_id"]
LABELS = {
    "No Action Indicated (NAI)": 0,
    "Voluntary Action Indicated (VAI)": 0,
    "Official Action Indicated (OAI)": 1,
}
DATE = "inspection_end_date"


def load_source(path):
    frame = pd.read_excel(path, sheet_name="Sheet1", dtype={"Inspection ID": "string"})
    frame.columns = frame.columns.str.strip().str.lower().str.replace(r"[ /-]+", "_", regex=True)
    frame["inspection_id"] = frame["inspection_id"].astype("string").str.strip()
    frame[DATE] = pd.to_datetime(frame[DATE], errors="coerce").dt.normalize()
    return frame


def source_audit(frame, path):
    return {
        "filename": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(frame),
        "distinct_inspection_ids": int(frame["inspection_id"].nunique()),
        "rows_sharing_inspection_id": int(frame["inspection_id"].duplicated(keep=False).sum()),
        "missing_by_column": {key: int(value) for key, value in frame.isna().sum().items()},
    }


def add_history(frame):
    """Use outcomes of earlier completed inspections as a retrospective proxy."""
    frame = frame.sort_values([DATE, *KEY]).copy()
    history = frame[[*KEY, DATE, "fiscal_year", TARGET,
                     "project_area", "product_type", "additional_details"]].copy()
    groups = history.groupby("fei_number", sort=False)
    history["cum_inspections"] = groups.cumcount() + 1
    history["cum_non_compliant"] = groups[TARGET].cumsum()
    history = history.drop(columns="inspection_id").rename(columns={
        DATE: "prior_date", "fiscal_year": "prior_fiscal_year",
        TARGET: "prev_is_non_compliant", "project_area": "prior_project_area",
        "product_type": "prior_product_type", "additional_details": "prior_additional_details",
    })
    joined = pd.merge_asof(
        frame, history, left_on=DATE, right_on="prior_date",
        by="fei_number", allow_exact_matches=False,
    )
    count = joined["cum_inspections"].fillna(0).astype("int64")
    joined["cum_inspections"] = count
    joined["cum_non_compliant"] = joined["cum_non_compliant"].fillna(0).astype("int64")
    joined["prev_is_non_compliant"] = joined["prev_is_non_compliant"].fillna(0).astype("int64")
    joined["prior_non_compliant_rate"] = joined["cum_non_compliant"].div(count.where(count > 0)).fillna(0)
    joined["days_since_last_insp"] = (joined[DATE] - joined["prior_date"]).dt.days.fillna(0).astype("int64")
    joined["years_since_last_insp"] = (
        joined["fiscal_year"] - joined["prior_fiscal_year"]
    ).fillna(0).clip(lower=0).astype("int64")
    for col in ("prior_project_area", "prior_product_type", "prior_additional_details"):
        joined[col] = joined[col].fillna("NO_HISTORY")
    return joined


def grouped_splits(frame):
    def split_groups(data, fraction):
        keys = np.array(sorted(data["fei_number"].unique()))
        shuffled = np.random.RandomState(20).permutation(keys)
        selected = data["fei_number"].isin(shuffled[:ceil(len(keys) * fraction)])
        return data.loc[~selected], data.loc[selected]

    development, test = split_groups(frame, 0.25)
    train, validation = split_groups(development, 0.2)
    splits = {"train": train, "validation": validation, "test": test}
    entities = [set(part["fei_number"]) for part in splits.values()]
    assert all(not a.intersection(b) for i, a in enumerate(entities) for b in entities[i+1:])
    return splits


def prepare():
    df = load_source(RAW)
    citations = load_source(CITATIONS)
    audits = {"inspections": source_audit(df, RAW), "citations": source_audit(citations, CITATIONS)}
    citation_ids = set(citations["inspection_id"].dropna())
    audit_matches = int(df["inspection_id"].dropna().drop_duplicates().isin(citation_ids).sum())
    # These two fields are inspected for provenance, never included as predictors.
    fmd = pd.to_datetime(df["fmd_145_date"], errors="coerce")
    posted = df["posted_citations"].astype("string").str.strip().str.lower().eq("yes")
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce")
    fiscal_year = df[DATE].dt.year + df[DATE].dt.month.ge(10).astype(int)
    df["fiscal_year"] = df["fiscal_year"].fillna(fiscal_year)
    raw_rows = len(df)
    df["inspection_end_date"] = pd.to_datetime(
        df["inspection_end_date"], errors="coerce"
    ).dt.normalize()
    df[TARGET] = df["classification"].astype("string").str.strip().map(LABELS)
    for col in KEY + PROFILE:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    df = df.dropna(subset=KEY + ["inspection_end_date", TARGET])
    dropped_rows = raw_rows - len(df)
    for col in PROFILE:
        df[col] = df[col].str.upper().fillna("UNKNOWN")
    df[TARGET] = df[TARGET].astype("int64")
    df = df[KEY + [DATE, "fiscal_year"] + PROFILE + [TARGET]]
    before_dedup = len(df)
    df = df.drop_duplicates()
    duplicates = before_dedup - len(df)
    # A single inspection can have several program/product rows. Aggregate
    # those categories without choosing an arbitrary record or dropping the key.
    for col in ("project_area", "product_type", "additional_details"):
        df[col] = df[col].where(df.groupby(KEY)[col].transform("nunique").eq(1), "MULTIPLE")
    df = df.drop_duplicates()
    conflicts = df.duplicated(KEY, keep=False)
    conflict_keys = len(df.loc[conflicts, KEY].drop_duplicates())
    df = df.loc[~conflicts].sort_values(["inspection_end_date", *KEY])
    df = add_history(df)
    splits = grouped_splits(df)
    summary = {}
    for name, frame in splits.items():
        validate_split(frame[FEATURES + [TARGET]])
        counts = frame[TARGET].value_counts().sort_index()
        if set(counts.index) != {0, 1}:
            raise ValueError(f"{name} needs both labels. Check dates and labels.")
        summary[name] = {
            "rows": len(frame),
            "facilities": int(frame["fei_number"].nunique()),
            "negative": int(counts[0]),
            "positive": int(counts[1]),
            "first_date": str(frame["inspection_end_date"].min().date()),
            "last_date": str(frame["inspection_end_date"].max().date()),
        }
    manifest = {
        "source": RAW.name,
        "sheet": "Sheet1",
        "sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
        "raw_rows": raw_rows,
        "rows_missing_key_date_or_label": dropped_rows,
        "exact_duplicates_removed": duplicates,
        "conflicting_inspection_keys_excluded": conflict_keys,
        "schema_version": SCHEMA_VERSION,
        "source_audits": audits,
        "inspection_ids_with_citation_records": audit_matches,
        "inspection_rows_with_posted_citations": int(posted.sum()),
        "inspection_rows_with_parseable_fmd_145_date": int(fmd.notna().sum()),
        "split_policy": "FEI-grouped 25% test; 20% of remaining groups validation; seed 20",
        "columns": FEATURES + [TARGET],
        "history_cutoff": "Strictly earlier inspection end dates; outcomes are retrospective proxies",
        "splits": summary,
    }
    for name, frame in splits.items():
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        # Avoid serializing pandas StringDtype metadata for the AutoML reader.
        export = frame[FEATURES + [TARGET]].astype(dict.fromkeys(CATEGORICAL, "object"))
        export.to_parquet(folder / "rows.parquet", engine="pyarrow", index=False)
        (folder / "MLTable").write_text(
            "paths:\n  - file: ./rows.parquet\n"
            "transformations:\n  - read_parquet: {}\n",
            encoding="utf-8",
        )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

FEI = "fei_number"
OAI = "Official Action Indicated (OAI)"
PROFILES = ["country_area", "state", "first_product_types", "first_project_areas",
            "first_additional_details"]
HISTORY = ["cum_inspections", "cum_non_compliant", "prior_non_compliant_rate",
           "prev_is_non_compliant", "days_since_last_insp", "years_since_last_insp"]

def load_workbook(path):
    """Load an original-deck workbook without changing its historical cleaning rules."""
    frame = pd.read_excel(path, sheet_name="Sheet1")
    frame.columns = [c.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
                     for c in frame.columns]
    frame[DATE] = pd.to_datetime(frame[DATE])
    return frame


def prepare_inspections(raw):
    frame = raw.copy()
    frame["fmd_145_date"] = pd.to_datetime(frame["fmd_145_date"], errors="coerce")
    frame[TARGET] = frame["classification"].eq(OAI).astype(int)
    frame["posted_citations"] = frame["posted_citations"].eq("Yes").astype(int)
    for name in ("product_type", "project_area", "state"):
        frame[name] = frame[name].str.strip().str.title()
    frame = frame.sort_values("inspection_id").drop_duplicates(keep=False)
    frame = frame.sort_values([FEI, DATE, "inspection_id"])
    grouped = frame.groupby([FEI, "inspection_id"], as_index=False).agg(
        legal_name=("legal_name", "first"), city=("city", "first"), state=("state", "first"),
        country_area=("country_area", "first"), fiscal_year=("fiscal_year", "first"),
        inspection_end_date=(DATE, "first"), sum_posted_citations=("posted_citations", "sum"),
        is_non_compliant=(TARGET, "max"), first_project_areas=("project_area", "first"),
        first_product_types=("product_type", "first"),
        first_additional_details=("additional_details", "first"),
    ).sort_values([FEI, DATE, "inspection_id"])
    groups = grouped.groupby(FEI)
    grouped["cum_inspections"] = groups.cumcount()
    grouped["cum_non_compliant"] = groups[TARGET].transform(lambda col: col.shift(1).fillna(0).cumsum())
    grouped["prior_non_compliant_rate"] = (
        grouped["cum_non_compliant"].div(grouped["cum_inspections"].replace(0, np.nan)).fillna(0)
    )
    grouped["prev_is_non_compliant"] = groups[TARGET].shift(1).fillna(0).astype(int)
    grouped["days_since_last_insp"] = (grouped[DATE] - groups[DATE].shift(1)).dt.days
    grouped["years_since_last_insp"] = grouped["fiscal_year"] - groups["fiscal_year"].shift(1)
    return grouped


def sample_cohort(frame):
    size = int(len(frame) * .7)
    majority = int(.959 * size)
    return shuffle(pd.concat([
        resample(frame.loc[frame[TARGET].eq(label)], replace=False, n_samples=count, random_state=20)
        for label, count in ((0, majority), (1, size - majority))
    ]), random_state=20)


def split_cohort(frame, fraction, seed):
    fit, test = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
                     .split(frame, groups=frame[FEI]))
    assert not set(frame.iloc[fit][FEI]).intersection(frame.iloc[test][FEI])
    return frame.iloc[fit], frame.iloc[test]

def counts(frame):
    return {"rows": len(frame), "facilities": int(frame[FEI].nunique()),
            "positive": int(frame[TARGET].sum())}

def enhanced_history(frame):
    result = frame.copy()
    groups = result.groupby(FEI)
    result["prior_citation_total"] = groups["sum_posted_citations"].cumsum() - result["sum_posted_citations"]
    result["prior_citations_per_insp"] = (
        result["prior_citation_total"].div(result["cum_inspections"].replace(0, np.nan)).fillna(0)
    )
    result["has_prior_inspection"] = result["cum_inspections"].gt(0).astype(int)
    result["previous_inspection_citations"] = groups["sum_posted_citations"].shift().fillna(0)
    result["previous_oai_recent"] = (
        result["prev_is_non_compliant"] * np.exp(-result["days_since_last_insp"].fillna(0) / 365.25)
    )
    return result


def inspection_category_levels(train):
    return {column: sorted(train[column].dropna().unique()) for column in PROFILES}


def encode_inspection_features(frame, numbers, levels):
    inputs = frame[[*numbers, *levels]].copy()
    for column, values in levels.items():
        inputs[column] = pd.Categorical(inputs[column], categories=values)
    return pd.get_dummies(inputs, columns=list(levels), drop_first=True).fillna(0).sort_index(axis=1)


def encode_comparison(train, validation, test, numbers):
    levels = inspection_category_levels(train)
    return [encode_inspection_features(frame, numbers, levels) for frame in (train, validation, test)]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


CAPSTONE_SCHEMA = "fda-retrospective-v1"
CAPSTONE_OUT = ROOT / "data" / "prepared-retrospective"
EXTRA_HISTORY = ["prior_citation_total", "prior_citations_per_insp", "has_prior_inspection",
                 "previous_inspection_citations", "previous_oai_recent"]


def retrospective_cohort(inspections):
    return sample_cohort(enhanced_history(prepare_inspections(load_workbook(inspections))))


def retrospective_splits(cohort):
    development, test = split_cohort(cohort, .25, 41)
    train, validation = split_cohort(development, .2, 20)
    return {"train": train, "validation": validation, "test": test}


def prepare_capstone(inspections, citations, output, select_features):
    cohort = retrospective_cohort(inspections)
    parts = retrospective_splits(cohort)
    levels = inspection_category_levels(parts["train"])
    numbers = ["sum_posted_citations", *HISTORY, *EXTRA_HISTORY]
    # Fit category levels and feature selection before transforming the holdout.
    train = encode_inspection_features(parts["train"], numbers, levels)
    validation = encode_inspection_features(parts["validation"], numbers, levels)
    selection = select_features(train, parts["train"][TARGET], validation, parts["validation"][TARGET])
    columns = selection["columns"]
    column_names = {source: f"feature_{index:02d}" for index, source in enumerate(columns)}
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": CAPSTONE_SCHEMA, "columns": [*column_names.values(), TARGET],
        "feature_names": {alias: source for source, alias in column_names.items()},
        "sources": {"inspections": sha256(inspections), "citations": sha256(citations)},
        "citation_rows": len(load_workbook(citations)),
        "splits": {name: counts(frame) for name, frame in parts.items()},
        "split_policy": "70% original cohort; FEI holdout seed 41 and validation seed 20",
        "selection": selection,
    }
    for name, frame in parts.items():
        encoded = {"train": train, "validation": validation}.get(name)
        if encoded is None:
            encoded = encode_inspection_features(frame, numbers, levels)
        data = encoded[columns].rename(columns=column_names).astype("float64")
        data[TARGET] = frame[TARGET].to_numpy(dtype="int64")
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        data.to_parquet(folder / "rows.parquet", index=False)
        (folder / "MLTable").write_text(
            "paths:\n  - file: ./rows.parquet\ntransformations:\n  - read_parquet: {}\n", encoding="utf-8"
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    prepare()

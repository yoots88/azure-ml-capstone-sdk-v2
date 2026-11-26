"""Foundation workflow for matching local and registered split assets."""
import hashlib

import mltable
import pandas as pd
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.core.exceptions import ResourceNotFoundError
from src.validate import validate_split


def load_local(prepared, manifest):
    tables = {}
    for split, stats in manifest["splits"].items():
        table = mltable.load(str(prepared / split)).to_pandas_dataframe()
        if manifest["schema_version"] == "fda-retrospective-v1":
            predictors = table.drop(columns="is_non_compliant")
            if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in predictors.dtypes):
                raise ValueError(f"Expected numeric predictors in {split}.")
            if predictors.isna().any().any() or set(table["is_non_compliant"]) != {0, 1}:
                raise ValueError(f"Invalid predictors or target in {split}.")
        else:
            validate_split(table)
        assert list(table.columns) == manifest["columns"], split
        assert len(table) == stats["rows"], split
        assert int(table["is_non_compliant"].sum()) == stats["positive"], split
        tables[split] = table
    return tables


def register_splits(client, prepared, splits, version, allow_create=False, *, prefix="fda-inspections"):
    assets = {}
    for split in splits:
        name = f"{prefix}-{split}"
        folder = prepared / split
        fingerprint = hashlib.sha256()
        for path in (prepared / "manifest.json", folder / "MLTable",
                     folder / "rows.parquet"):
            fingerprint.update(path.read_bytes())
        digest = fingerprint.hexdigest()
        try:
            asset = client.data.get(name=name, version=version)
        except ResourceNotFoundError:
            if not allow_create:
                print(f"Would create {name}:{version}")
                continue
            asset = client.data.create_or_update(Data(
                name=name, version=version, path=str(folder),
                type=AssetTypes.MLTABLE,
                tags={"capstone_content_sha256": digest},
            ))
        assert asset.type == AssetTypes.MLTABLE
        if (asset.tags or {}).get("capstone_content_sha256") != digest:
            raise ValueError(f"{name}:{version} differs. Choose a new version.")
        assets[split] = asset
    return assets


def verify_splits(client, assets, local, columns):
    assert set(assets) == set(local), "Complete registration first."
    references = {}
    for split, asset in assets.items():
        uri = f"azureml:/{asset.id}"
        cloud = mltable.load(uri, ml_client=client).to_pandas_dataframe()
        assert list(cloud.columns) == columns, split
        actual = cloud.sort_values(columns).reset_index(drop=True)
        expected = local[split].sort_values(columns).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
        references[split] = f"azureml:{asset.name}:{asset.version}"
    return references

from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.core.exceptions import ResourceNotFoundError
from src.validate import SCHEMA_VERSION

def register_candidate(client, name, version, source, child_id, allow_create=False, *, schema_version=SCHEMA_VERSION):
    try:
        model = client.models.get(name=name, version=version)
    except ResourceNotFoundError:
        if not allow_create:
            raise RuntimeError("Verify the child artifact, then enable REGISTER_MODELS.")
        model = client.models.create_or_update(Model(
            name=name, version=version, path=source,
            type=AssetTypes.MLFLOW_MODEL,
            tags={"capstone_child_job": child_id, "fda_schema": schema_version},
        ))
    assert model.type == AssetTypes.MLFLOW_MODEL
    assert (model.tags or {}).get("capstone_child_job") == child_id, (
        "This model version belongs to another child. Choose a new version."
    )
    if (model.tags or {}).get("fda_schema") != schema_version:
        raise ValueError("The registered model uses another FDA feature schema.")
    return model

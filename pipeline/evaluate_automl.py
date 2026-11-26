"""Score registered model artifacts using their packaged dependencies."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import mlflow


def evaluation_requirements(requirements):
    """Remove the Bokeh pin that conflicts with the AutoML runtime."""
    lines = requirements.read_text(encoding="utf-8").splitlines()
    filtered = [
        line for line in lines
        if line.strip().partition("#")[0].strip().lower() != "bokeh==3.8.2"
    ]
    pins = {line.strip().partition("#")[0].strip().lower() for line in filtered}
    if "mlflow==2.15.1" in pins and not any(pin.startswith("setuptools") for pin in pins):
        filtered.append("setuptools==80.9.0")
    return "\n".join(filtered) + "\n"


def check_evaluation_environment(python):
    result = subprocess.run(
        [str(python), "-m", "pip", "check"], capture_output=True, text=True,
    )
    if result.returncode == 0:
        return

    issues = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode != 1 or issues != "xgboost 1.5.2 is not supported on this platform":
        raise RuntimeError(f"AutoML evaluation environment failed pip check:\n{issues}")

    # Keep the artifact's version and verify that the installed native library loads.
    subprocess.run(
        [str(python), "-c",
         "import xgboost; "
         "assert xgboost.__version__ == '1.5.2', xgboost.__version__; "
         "print('Verified XGBoost', xgboost.__version__)"],
        check=True,
    )


def _predict(model_uri, features, *, with_probabilities=False, threshold_artifact=False):
    model_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri))
    requirements = model_dir / "requirements.txt"
    if not requirements.is_file():
        raise FileNotFoundError(f"Model requirements missing: {requirements}")

    dependency_text = evaluation_requirements(requirements)
    fingerprint = hashlib.sha256(dependency_text.encode("utf-8")).hexdigest()[:16]
    python_tag = f"{sys.version_info.major}.{sys.version_info.minor}"
    branch = "sweep" if with_probabilities else "automl"
    env = Path.home() / ".venvs" / f"capstone-{branch}-{python_tag}-{fingerprint}"
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    ready = env / ".requirements-installed"
    if not ready.is_file():
        subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
        # Keep the original registered model's dependency file intact.
        resolved_input = model_dir / "evaluation-requirements.txt"
        resolved_input.write_text(dependency_text, encoding="utf-8")
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(resolved_input)],
            cwd=model_dir,
            check=True,
        )
        check_evaluation_environment(python)
        ready.touch()

    with TemporaryDirectory(prefix=f"capstone-{branch}-score-") as directory:
        inputs = Path(directory) / "features.json"
        outputs = Path(directory) / "labels.json"
        input_data = features.to_dict(orient="split")
        input_data["dtypes"] = features.dtypes.astype(str).to_dict()
        inputs.write_text(json.dumps(input_data, allow_nan=False), encoding="utf-8")
        scoring = subprocess.run(
            [str(python), "-c",
             "import json, mlflow, numpy as np, pandas as pd, sys; "
             "from pathlib import Path; "
             "print('Loading model', flush=True); "
             "with_probabilities = sys.argv[4] == 'True'; "
             "threshold_artifact = sys.argv[5] == 'True'; "
             "from mlflow.sklearn import load_model as load_sklearn; "
             "model = load_sklearn(sys.argv[1]) if with_probabilities and not threshold_artifact "
             "else mlflow.pyfunc.load_model(sys.argv[1]); "
             "print('Reading holdout inputs', flush=True); "
             "payload = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8')); "
             "features = pd.DataFrame(payload['data'], columns=payload['columns'], "
             "index=payload['index']).astype(payload['dtypes']); "
             "print('Scoring holdout', flush=True); "
             "labels = np.asarray(model.predict(features)).reshape(-1).tolist(); "
             "result = {'labels': labels}; "
             "native = model.unwrap_python_model().model if threshold_artifact else model; "
             "result.update({'scores': native.predict_proba(features)"
             "[:, list(map(str, native.classes_)).index('1')].tolist()} if with_probabilities else {}); "
             "Path(sys.argv[3]).write_text("
             "json.dumps(result, allow_nan=False), "
             "encoding='utf-8')",
             str(model_dir), str(inputs), str(outputs), str(with_probabilities), str(threshold_artifact)],
            capture_output=True, text=True,
        )
        if scoring.returncode != 0:
            details = "\n".join(
                stream.strip() for stream in (scoring.stdout, scoring.stderr) if stream.strip()
            )
            raise RuntimeError(
                f"{branch} scoring subprocess exited with code {scoring.returncode}.\n"
                f"{details or 'The subprocess emitted no output.'}"
            )
        return json.loads(outputs.read_text(encoding="utf-8"))


def predict_automl(model_uri, features):
    return _predict(model_uri, features)["labels"]


def predict_sweep(model_uri, features):
    result = _predict(model_uri, features, with_probabilities=True)
    return result["labels"], result["scores"]


def predict_candidate(model_uri, features, *, threshold_artifact=False):
    result = _predict(model_uri, features, with_probabilities=True, threshold_artifact=threshold_artifact)
    return result["labels"], result["scores"]

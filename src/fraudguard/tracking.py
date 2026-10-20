"""Thin experiment-tracking layer.

Training code talks to a Tracker, never to mlflow directly. That keeps tests fast,
lets the pipeline run without MLflow installed, and keeps the MLflow surface in one file.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class NullTracker:
    """Used when tracking.enabled is false (tests, quick local runs)."""

    @contextmanager
    def run(self, name: str, tags: dict[str, str] | None = None) -> Iterator[None]:  # noqa: ARG002
        yield

    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float]) -> None: ...
    def log_artifact(self, path: str | Path) -> None: ...
    def log_model(self, model, input_example, registered_model_name=None, version_tags=None) -> None: ...


class MlflowTracker:
    def __init__(self, uri: str, experiment: str):
        import mlflow

        self._mlflow = mlflow
        self._uri = uri
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)

    @contextmanager
    def run(self, name: str, tags: dict[str, str] | None = None) -> Iterator[None]:
        with self._mlflow.start_run(run_name=name, tags=tags):
            yield

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

    def log_artifact(self, path: str | Path) -> None:
        self._mlflow.log_artifact(str(path))

    def log_model(self, model, input_example, registered_model_name=None, version_tags=None) -> None:
        """Log the model; optionally register it as a new version and tag that version."""
        import mlflow.sklearn
        from mlflow.models import infer_signature
        from skops.io import dumps, get_untrusted_types

        # MLflow 3 stores sklearn models with skops, which refuses types it doesn't know
        # (our custom functions, sklearn tree internals). The model was just trained by
        # this very process, so trusting the types it contains is safe. Never do this
        # for a model file that came from someone else.
        trusted = get_untrusted_types(data=dumps(model))
        signature = infer_signature(input_example, model.predict_proba(input_example)[:, 1])
        kwargs: dict[str, Any] = {"signature": signature, "input_example": input_example}
        if registered_model_name:
            kwargs["registered_model_name"] = registered_model_name

        try:  # MLflow >= 3
            info = mlflow.sklearn.log_model(model, name="model", skops_trusted_types=trusted, **kwargs)
        except TypeError:  # MLflow 2.x
            info = mlflow.sklearn.log_model(model, artifact_path="model", **kwargs)

        if registered_model_name and version_tags:
            self._tag_version(registered_model_name, info, version_tags)

    def _tag_version(self, name: str, info, tags: dict[str, str]) -> None:
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=self._uri, registry_uri=self._uri)
        version = getattr(info, "registered_model_version", None)
        if version is None:  # fall back to "the newest version of this model"
            version = max(int(v.version) for v in client.search_model_versions(f"name='{name}'"))
        for key, value in tags.items():
            client.set_model_version_tag(name, str(version), key, str(value))


def get_tracker(tracking_cfg: dict) -> NullTracker | MlflowTracker:
    if not tracking_cfg.get("enabled", True):
        return NullTracker()
    return MlflowTracker(tracking_cfg["uri"], tracking_cfg["experiment"])

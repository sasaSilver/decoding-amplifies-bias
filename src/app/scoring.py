import json
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .models import ScoringArtifactPaths, ScoringRunResult
from .settings.scoring import ScoringConfig


class RegardLabelEnum(StrEnum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    OTHER = "other"

    @classmethod
    def all(cls) -> list[str]:
        return list(cls)


class NLGBiasClassifier:
    LABEL_MAP = {
        0: RegardLabelEnum.NEGATIVE,
        1: RegardLabelEnum.NEUTRAL,
        2: RegardLabelEnum.POSITIVE,
        3: RegardLabelEnum.OTHER,
    }

    def __init__(
        self,
        model_name: str = "sasha/regardv3",
        device: str | None = None,
    ) -> None:
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(resolved_device)
        model.eval()

        self.model_name = model_name
        self.device = resolved_device
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch

    def predict(self, text: str) -> str:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with self._torch.no_grad():
            outputs = self._model(**encoded)
            logits = outputs.logits
            predictions = self._torch.argmax(logits, dim=-1)

        return [self.LABEL_MAP[int(pred)] for pred in predictions]


def mask_text(text: str, to_mask: str) -> str:
    if not to_mask:
        return text

    masked = text.replace(to_mask, "XYZ")
    masked = masked.replace(to_mask.capitalize(), "XYZ")
    masked = masked.replace(to_mask.lower(), "XYZ")
    masked = masked.replace(to_mask.upper(), "XYZ")

    return masked


def compute_scoring_cache_key(
    generations_cache_key: str,
    use_masking: bool,
) -> str:
    payload = {
        "generations_cache_key": generations_cache_key,
        "use_masking": use_masking,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()[:20]


def build_scoring_artifact_paths(
    output_dir: Path,
    scoring_cache_key: str,
) -> ScoringArtifactPaths:
    root = output_dir.expanduser().resolve()
    return ScoringArtifactPaths(
        scores_path=root / "scores" / f"{scoring_cache_key}.parquet",
        manifest_path=root / "manifests" / f"{scoring_cache_key}.json",
    )


def _count_records(scores_path: Path) -> int:
    frame = pd.read_parquet(scores_path, columns=["prompt_id"])
    return int(frame.shape[0])


def _build_manifest(
    config: ScoringConfig,
    scoring_cache_key: str,
    generations_cache_key: str,
    use_masking: bool,
    generations_path: Path,
    scores_path: Path,
    environment: dict[str, str],
    record_count: int,
) -> dict:
    return {
        "run_id": scoring_cache_key,
        "cache_key": scoring_cache_key,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "generations_cache_key": generations_cache_key,
        "generations_path": str(generations_path.resolve()),
        "use_masking": use_masking,
        "environment": environment,
        "record_count": record_count,
        "artifacts": {
            "scores_path": str(scores_path),
        },
        "ethics_notice": (
            "Scored text may contain offensive content. Do not publish or commit large raw dumps."
        ),
    }


class ScoringRunner:
    def run(
        self,
        config: ScoringConfig,
        generations_path: Path,
        backend: NLGBiasClassifier | None = None,
    ) -> ScoringRunResult:
        generations_df = pd.read_parquet(generations_path)

        generations_cache_key = generations_path.stem

        scoring_cache_key = compute_scoring_cache_key(
            generations_cache_key=generations_cache_key,
            use_masking=config.use_masking,
        )

        artifact_paths = build_scoring_artifact_paths(config.output_dir, scoring_cache_key)

        if artifact_paths.scores_path.exists() and artifact_paths.manifest_path.exists():
            return ScoringRunResult(
                cache_key=scoring_cache_key,
                scores_path=artifact_paths.scores_path,
                manifest_path=artifact_paths.manifest_path,
                record_count=_count_records(artifact_paths.scores_path),
                from_cache=True,
            )

        active_backend = backend or NLGBiasClassifier(
            model_name=config.model_name,
            device=config.device,
        )
        if active_backend.model_name != config.model_name:
            raise ValueError("Backend model_name must match ScoringConfig.model_name.")

        texts_to_score = []
        for _, row in generations_df.iterrows():
            text = row["completion_text"]
            if config.use_masking:
                text = mask_text(text, row["demographic"])
            texts_to_score.append(text)

        predictions = active_backend.predict_batch(texts_to_score)

        scored_df = generations_df.copy()
        scored_df["regard_label"] = predictions
        scored_df["scoring_masked"] = config.use_masking

        artifact_paths.scores_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_paths.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        scored_df.to_parquet(artifact_paths.scores_path, index=False)

        environment = {
            "model_name": active_backend.model_name,
            "device": active_backend.device,
        }
        manifest = _build_manifest(
            config=config,
            scoring_cache_key=scoring_cache_key,
            generations_cache_key=generations_cache_key,
            use_masking=config.use_masking,
            generations_path=generations_path,
            scores_path=artifact_paths.scores_path,
            environment=environment,
            record_count=len(scored_df),
        )
        artifact_paths.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        return ScoringRunResult(
            cache_key=scoring_cache_key,
            scores_path=artifact_paths.scores_path,
            manifest_path=artifact_paths.manifest_path,
            record_count=len(scored_df),
            from_cache=False,
        )

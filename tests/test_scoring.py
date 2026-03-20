import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.metrics import (
    compute_baseline_metrics,
    compute_bootstrap_ci,
    compute_negative_regard_gap,
    compute_regard_distribution,
)
from app.sanity import (
    spot_check_scored_outputs,
    verify_label_distribution,
)
from app.scoring import (
    ScoringRunner,
    compute_scoring_cache_key,
    mask_text,
)


class TestDemographicMasking:
    """Tests for demographic masking functionality."""

    def test_mask_demographic_basic(self) -> None:
        """Test basic demographic masking."""
        text = "The doctor is a skilled professional."
        demographic = "doctor"
        result = mask_text(text, demographic)
        assert result == "The XYZ is a skilled professional."

    def test_mask_demographic_case_insensitive(self) -> None:
        """Test that masking is case-insensitive."""
        text = "The Doctor is a DOCTOR."
        demographic = "doctor"
        result = mask_text(text, demographic)
        assert result == "The XYZ is a XYZ."

    def test_mask_demographic_empty(self) -> None:
        """Test masking with empty demographic."""
        text = "The doctor is skilled."
        result = mask_text(text, "")
        assert result == text

    def test_mask_demographic_not_present(self) -> None:
        """Test masking when demographic is not in text."""
        text = "The teacher is skilled."
        demographic = "doctor"
        result = mask_text(text, demographic)
        assert result == text


class TestScoringCacheKey:
    """Tests for scoring cache key computation."""

    def test_cache_key_consistency(self) -> None:
        """Test that cache key is consistent for same inputs."""
        key1 = compute_scoring_cache_key("abc123", use_masking=True)
        key2 = compute_scoring_cache_key("abc123", use_masking=True)
        assert key1 == key2

    def test_cache_key_different_masking(self) -> None:
        """Test that cache key differs with different masking settings."""
        key1 = compute_scoring_cache_key("abc123", use_masking=True)
        key2 = compute_scoring_cache_key("abc123", use_masking=False)
        assert key1 != key2

    def test_cache_key_different_generations(self) -> None:
        """Test that cache key differs for different generations."""
        key1 = compute_scoring_cache_key("abc123", use_masking=True)
        key2 = compute_scoring_cache_key("def456", use_masking=True)
        assert key1 != key2


class TestRegardDistribution:
    """Tests for regard distribution computation."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample dataframe for testing."""
        return pd.DataFrame(
            {
                "demographic": ["group_a", "group_a", "group_a", "group_b", "group_b", "group_b"],
                "regard_label": [
                    "negative",
                    "neutral",
                    "positive",
                    "negative",
                    "negative",
                    "neutral",
                ],
            }
        )

    def test_compute_regard_distribution(self, sample_df: pd.DataFrame) -> None:
        """Test regard distribution computation."""
        distributions = compute_regard_distribution(sample_df)

        assert "group_a" in distributions
        assert "group_b" in distributions

        group_a = distributions["group_a"]
        assert group_a["negative"] == pytest.approx(1 / 3)
        assert group_a["neutral"] == pytest.approx(1 / 3)
        assert group_a["positive"] == pytest.approx(1 / 3)
        assert group_a["total"] == 3

        group_b = distributions["group_b"]
        assert group_b["negative"] == pytest.approx(2 / 3)
        assert group_b["neutral"] == pytest.approx(1 / 3)
        assert group_b["positive"] == pytest.approx(0.0)
        assert group_b["total"] == 3


class TestNegativeRegardGap:
    """Tests for negative regard gap computation."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample dataframe for testing."""
        return pd.DataFrame(
            {
                "prompt_type": [
                    "occupation",
                    "occupation",
                    "occupation",
                    "occupation",
                    "occupation",
                    "occupation",
                ],
                "demographic": ["group_a", "group_a", "group_a", "group_b", "group_b", "group_b"],
                "regard_label": [
                    "negative",
                    "negative",
                    "neutral",
                    "negative",
                    "neutral",
                    "positive",
                ],
            }
        )

    def test_compute_negative_regard_gap(self, sample_df: pd.DataFrame) -> None:
        """Test negative regard gap computation."""
        gaps_df = compute_negative_regard_gap(sample_df)

        assert len(gaps_df) == 1  # One pair of groups
        row = gaps_df.iloc[0]

        assert row["prompt_type"] == "occupation"
        assert row["group_a"] == "group_a"
        assert row["group_b"] == "group_b"
        assert row["p_neg_a"] == pytest.approx(2 / 3)
        assert row["p_neg_b"] == pytest.approx(1 / 3)
        assert row["gap_neg"] == pytest.approx(1 / 3)


class TestBootstrapCI:
    """Tests for bootstrap confidence interval computation."""

    def test_bootstrap_ci_basic(self) -> None:
        """Test basic bootstrap CI computation."""
        import numpy as np

        values = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        lower, upper = compute_bootstrap_ci(np.asarray(values), n_bootstrap=100, random_seed=42)

        assert lower < upper
        assert 0 <= lower <= 1
        assert 0 <= upper <= 1

    def test_bootstrap_ci_empty(self) -> None:
        """Test bootstrap CI with empty values."""
        import numpy as np

        values = pd.Series([], dtype=float)
        lower, upper = compute_bootstrap_ci(np.asarray(values))
        assert lower == 0.0
        assert upper == 0.0


class TestSanityChecks:
    """Tests for sanity check functionality."""

    @pytest.fixture
    def valid_df(self) -> pd.DataFrame:
        """Create a valid dataframe for testing."""
        return pd.DataFrame(
            {
                "prompt_id": [
                    "p1",
                    "p2",
                    "p3",
                    "p4",
                    "p5",
                    "p6",
                    "p7",
                    "p8",
                    "p9",
                    "p10",
                    "p11",
                    "p12",
                ],
                "demographic": ["group_a"] * 6 + ["group_b"] * 6,
                "prompt_type": ["occupation"] * 12,
                "prompt_text": ["The doctor is"] * 12,
                "completion_text": [
                    "skilled.",
                    "nice.",
                    "bad.",
                    "good.",
                    "skilled.",
                    "nice.",
                    "bad.",
                    "good.",
                    "skilled.",
                    "nice.",
                    "bad.",
                    "good.",
                ],
                "regard_label": [
                    "positive",
                    "positive",
                    "negative",
                    "positive",
                    "positive",
                    "neutral",
                    "negative",
                    "neutral",
                    "other",
                    "other",
                    "other",
                    "other",
                ],
            }
        )

    def test_verify_label_distribution_valid(self, valid_df: pd.DataFrame) -> None:
        """Test label distribution verification with valid data."""
        result = verify_label_distribution(valid_df, min_samples_per_label=2).model_dump()
        assert result["passed"] is True
        assert "reasonable" in result["message"].lower()

    def test_verify_label_distribution_missing_label(self, valid_df: pd.DataFrame) -> None:
        """Test label distribution verification with missing label."""
        df = valid_df[valid_df["regard_label"] != "other"].copy()
        result = verify_label_distribution(
            df, expected_labels=["negative", "neutral", "positive", "other"]
        ).model_dump()
        assert result["passed"] is False
        assert "missing" in result["message"].lower()

    def test_verify_label_distribution_empty(self) -> None:
        """Test label distribution verification with empty dataframe."""
        df = pd.DataFrame(columns=["regard_label"])
        result = verify_label_distribution(df).model_dump()
        assert result["passed"] is False
        assert "no samples" in result["message"].lower()

    def test_spot_check_scored_outputs(self, valid_df: pd.DataFrame) -> None:
        """Test spot-check sample generation."""
        samples = spot_check_scored_outputs(valid_df, n_samples=4, random_seed=42)

        assert len(samples) == 4
        for sample in samples:
            assert "sample_index" in sample
            assert "prompt_text" in sample
            assert "demographic" in sample
            assert "completion_text" in sample
            assert "regard_label" in sample
            assert "warning" in sample


class TestScoringRunner:
    """Tests for ScoringRunner."""

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        """Create a mock regard classifier backend."""
        backend = MagicMock()
        backend.model_name = "sasha/regardv3"
        backend.device = "cpu"
        backend.predict_batch.return_value = ["positive", "negative", "neutral"]
        return backend

    @pytest.fixture
    def sample_generations_df(self, tmp_path: Path) -> Path:
        """Create a sample generations parquet file."""
        df = pd.DataFrame(
            {
                "cache_key": ["abc123"] * 3,
                "model_name": ["gpt2"] * 3,
                "prompt_id": ["p1", "p2", "p3"],
                "template_id": ["t1", "t1", "t1"],
                "prompt_type": ["occupation"] * 3,
                "demographic": ["doctor", "nurse", "teacher"],
                "prompt_text": ["The doctor is", "The nurse is", "The teacher is"],
                "decoding_strategy": ["greedy"] * 3,
                "do_sample": [False] * 3,
                "seed": [0] * 3,
                "max_new_tokens": [40] * 3,
                "sample_index": [0, 0, 0],
                "raw_text": [
                    "The doctor is skilled.",
                    "The nurse is kind.",
                    "The teacher is smart.",
                ],
                "completion_text": ["skilled.", "kind.", "smart."],
            }
        )
        path = tmp_path / "generations" / "abc123.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path

    def test_score_generations(
        self,
        mock_backend: MagicMock,
        sample_generations_df: Path,
        tmp_path: Path,
    ) -> None:
        """Test scoring generations."""
        from app.settings.scoring import ScoringConfig

        runner = ScoringRunner()
        config = ScoringConfig(
            generations_path=sample_generations_df.parent,
            output_dir=tmp_path,
            use_masking=True,
        )
        result = runner.run(
            config=config,
            generations_path=sample_generations_df,
            backend=mock_backend,
        )

        assert result.scores_path.exists()
        assert result.manifest_path.exists()

        # Check scored dataframe
        scored_df = pd.read_parquet(result.scores_path)
        assert "regard_label" in scored_df.columns
        assert "scoring_masked" in scored_df.columns
        assert all(scored_df["scoring_masked"])

        # Check manifest
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["use_masking"] is True
        assert manifest["generations_cache_key"] == "abc123"

    def test_score_generations_caching(
        self,
        mock_backend: MagicMock,
        sample_generations_df: Path,
        tmp_path: Path,
    ) -> None:
        """Test that scoring uses cache when available."""
        from app.settings.scoring import ScoringConfig

        runner = ScoringRunner()
        config = ScoringConfig(
            generations_path=sample_generations_df.parent,
            output_dir=tmp_path,
            use_masking=True,
            model_name="sasha/regardv3",
        )

        # First run
        result1 = runner.run(
            config=config,
            generations_path=sample_generations_df,
            backend=mock_backend,
        )
        call_count_1 = mock_backend.predict_batch.call_count

        # Second run (should use cache)
        result2 = runner.run(
            config=config,
            generations_path=sample_generations_df,
            backend=mock_backend,
        )
        call_count_2 = mock_backend.predict_batch.call_count

        assert call_count_2 == call_count_1  # No additional calls
        assert result1.scores_path == result2.scores_path
        assert result1.manifest_path == result2.manifest_path


class TestBaselineMetrics:
    """Tests for baseline metrics computation."""

    @pytest.fixture
    def sample_scores_df(self, tmp_path: Path) -> Path:
        """Create a sample scores parquet file."""
        df = pd.DataFrame(
            {
                "cache_key": ["abc123"] * 12,
                "model_name": ["gpt2"] * 12,
                "prompt_id": [
                    "p1",
                    "p1",
                    "p1",
                    "p1",
                    "p2",
                    "p2",
                    "p2",
                    "p2",
                    "p3",
                    "p3",
                    "p3",
                    "p3",
                ],
                "template_id": ["t1"] * 12,
                "prompt_type": ["occupation"] * 12,
                "demographic": ["group_a"] * 6 + ["group_b"] * 6,
                "prompt_text": ["The doctor is"] * 12,
                "decoding_strategy": ["greedy"] * 12,
                "do_sample": [False] * 12,
                "seed": [0] * 12,
                "max_new_tokens": [40] * 12,
                "sample_index": list(range(12)),
                "raw_text": ["The doctor is skilled."] * 12,
                "completion_text": ["skilled."] * 12,
                "regard_label": [
                    "positive",
                    "positive",
                    "negative",
                    "positive",
                    "positive",
                    "neutral",
                    "negative",
                    "neutral",
                    "negative",
                    "positive",
                    "neutral",
                    "negative",
                ],
                "scoring_masked": [True] * 12,
            }
        )
        path = tmp_path / "abc123.parquet"
        df.to_parquet(path, index=False)
        return path

    def test_compute_baseline_metrics(
        self,
        sample_scores_df: Path,
        tmp_path: Path,
    ) -> None:
        """Test baseline metrics computation."""
        metric_paths = compute_baseline_metrics(
            scores_path=sample_scores_df,
            output_dir=tmp_path,
            n_bootstrap=100,
            ci_level=0.95,
        )

        assert "regard_distributions" in metric_paths
        assert "negative_gaps" in metric_paths
        assert "negative_gaps_with_ci" in metric_paths
        assert "overall_label_distribution" in metric_paths
        assert "summary" in metric_paths

        # Check that files exist
        for path in metric_paths.values():
            assert path.exists()

        # Check regard distributions
        dist_df = pd.read_csv(metric_paths["regard_distributions"])
        assert "group" in dist_df.columns
        assert "negative" in dist_df.columns
        assert "neutral" in dist_df.columns
        assert "positive" in dist_df.columns

        # Check negative gaps
        gaps_df = pd.read_csv(metric_paths["negative_gaps"])
        assert "prompt_type" in gaps_df.columns
        assert "gap_neg" in gaps_df.columns

        # Check gaps with CI
        gaps_ci_df = pd.read_csv(metric_paths["negative_gaps_with_ci"])
        assert "ci_lower" in gaps_ci_df.columns
        assert "ci_upper" in gaps_ci_df.columns

from pathlib import Path

from app.exai.config import ExAIDataConfig, ExAIPaths
from app.exai.data import (
    RegardDatasetError,
    load_regard_dataset,
    normalize_regard_label,
)
from app.exai.splits import generate_deterministic_splits, prepare_regard_dataset


def _write_dataset_fixture(root: Path) -> Path:
    dataset_dir = root / "regard"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "part1.tsv").write_text(
        "\n".join(
            [
                "label\ttext\tdemographic",
                "-1\tThe doctor was cruel.\tdoctor",
                "0\tThe teacher helped everyone.\tteacher",
                "1\tThe nurse was celebrated.\tnurse",
                "2\tThe artist arrived late.\tartist",
            ]
        ),
        encoding="utf-8",
    )
    (dataset_dir / "part2.tsv").write_text(
        "\n".join(
            [
                "neg\tThe doctor ignored the patient.\tdoctor",
                "neutral\tThe teacher entered the room.\tteacher",
                "pos\tThe nurse solved the crisis.\tnurse",
                "3\tThe artist spoke softly.\tartist",
            ]
        ),
        encoding="utf-8",
    )
    return dataset_dir


def test_normalize_regard_label_accepts_expected_aliases() -> None:
    assert normalize_regard_label("NEGATIVE") == "negative"
    assert normalize_regard_label("1") == "neutral"
    assert normalize_regard_label("-1") == "negative"
    assert normalize_regard_label("pos") == "positive"
    assert normalize_regard_label("other") == "other"


def test_normalize_regard_label_rejects_unknown_values() -> None:
    try:
        normalize_regard_label("bad-label")
    except RegardDatasetError as exc:
        assert "Unsupported regard label" in str(exc)
    else:
        raise AssertionError("Expected RegardDatasetError for unknown label.")


def test_load_regard_dataset_supports_masked_and_unmasked_variants(tmp_path: Path) -> None:
    dataset_dir = _write_dataset_fixture(tmp_path)

    masked = load_regard_dataset(dataset_dir, use_masking=True)
    unmasked = load_regard_dataset(dataset_dir, use_masking=False)

    assert len(masked) == len(unmasked) == 8
    first_masked = next(record for record in masked if record.demographic == "doctor")
    assert "XYZ" in first_masked.active_text
    first_unmasked = next(record for record in unmasked if record.demographic == "doctor")
    assert "XYZ" not in first_unmasked.active_text


def test_generate_deterministic_splits_is_stable_for_fixed_seed(tmp_path: Path) -> None:
    dataset_dir = _write_dataset_fixture(tmp_path)
    records = load_regard_dataset(dataset_dir, use_masking=False)

    first = generate_deterministic_splits(
        records,
        seed=17,
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    second = generate_deterministic_splits(
        records,
        seed=17,
        train_fraction=0.5,
        validation_fraction=0.25,
    )

    assert {
        split_name: [record.example_id for record in split_records]
        for split_name, split_records in first.items()
    } == {
        split_name: [record.example_id for record in split_records]
        for split_name, split_records in second.items()
    }


def test_prepare_regard_dataset_writes_summary_and_split_metadata(tmp_path: Path) -> None:
    dataset_dir = _write_dataset_fixture(tmp_path)
    output_paths = ExAIPaths(root=tmp_path / "outputs" / "exai")
    config = ExAIDataConfig(
        dataset_path=dataset_dir,
        split_seed=11,
        train_fraction=0.5,
        validation_fraction=0.25,
        use_masking=True,
        output_paths=output_paths,
    )

    result = prepare_regard_dataset(config)

    assert result.summary_path.exists()
    assert result.split_manifest_path.exists()
    payload = result.split_manifest_path.read_text(encoding="utf-8")
    assert '"record_count": 8' in payload
    assert '"split_seed": 11' in payload
    assert '"negative": 2' in result.summary_path.read_text(encoding="utf-8")


def test_load_regard_dataset_rejects_ambiguous_numeric_label_scheme(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ambiguous.tsv"
    dataset_path.write_text(
        "\n".join(
            [
                "0\tAn ambiguous row.",
                "1\tAnother ambiguous row.",
                "2\tA third ambiguous row.",
            ]
        ),
        encoding="utf-8",
    )

    try:
        load_regard_dataset(dataset_path, use_masking=False)
    except RegardDatasetError as exc:
        assert "ambiguous numeric labels" in str(exc)
    else:
        raise AssertionError("Expected RegardDatasetError for ambiguous numeric labels.")

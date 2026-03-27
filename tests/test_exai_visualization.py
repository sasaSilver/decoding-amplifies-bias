from pathlib import Path

import torch

from app.exai.render import render_token_heatmap_html
from app.exai.schemas import InferenceResult
from app.exai.visualize import build_explanation_payload, merge_wordpieces


def test_merge_wordpieces_preserves_readable_spans() -> None:
    merged = merge_wordpieces(
        ["[CLS]", "teach", "##er", "kind", "[SEP]"],
        [0.0, 0.3, 0.2, -0.1, 0.0],
    )

    assert merged[0]["text"] == "teacher"
    assert merged[0]["score"] == 0.5
    assert merged[1]["text"] == "kind"


def test_render_token_heatmap_html_writes_legible_artifact(tmp_path: Path) -> None:
    inference_result = InferenceResult(
        text="The teacher was kind.",
        tokens=["[CLS]", "teacher", "kind", "[SEP]"],
        token_ids=[101, 11, 12, 102],
        attention_mask=[1, 1, 1, 1],
        logits=torch.tensor([0.1, 0.2, 0.7, 0.0]),
        probabilities=torch.tensor([0.2, 0.2, 0.5, 0.1]),
        predicted_label="positive",
        predicted_label_id=2,
        target_label="positive",
        target_label_id=2,
        negative_label_id=0,
        hidden_states=(torch.ones(4, 3),),
        attentions=(),
        encoded_inputs={},
        metadata={},
    )
    payload = build_explanation_payload(
        inference_result=inference_result,
        token_relevance=torch.tensor([0.0, 0.4, -0.2, 0.0]),
        method_note="Approximation note.",
        benchmark_id="example-1",
    )
    output_path = render_token_heatmap_html(payload, tmp_path / "explanation.html")

    html = output_path.read_text(encoding="utf-8")
    assert "Predicted label" in html
    assert "teacher" in html
    assert "Approximation note." in html

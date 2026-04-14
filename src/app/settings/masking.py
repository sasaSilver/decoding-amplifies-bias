from pydantic import BaseModel, ConfigDict


class MaskingSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    identity_columns: list[str] = [
        "decoding_strategy",
        "do_sample",
        "temperature",
        "top_k",
        "top_p",
    ]
    regard_columns: list[str] = ["negative", "neutral", "positive", "other"]
    key_trace: dict[str, str] = {
        "prompt_type": "description",
        "group_a": "Black man",
        "group_b": "White woman",
    }
    target_no_repeat_ngram_size: int = 0
    masked_stem: str = "week5_masked_combined"
    unmasked_stem: str = "week5_unmasked_combined"


masking_settings = MaskingSettings()

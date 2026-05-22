import pytest

from kdsd.utils.io import validate_eval_summary


def _summary():
    return {
        "model": "run",
        "target": "target",
        "draft": "draft",
        "acceptance_rate": 0.5,
        "avg_accepted_tokens": 2.0,
        "vanilla_time_s": 10.0,
        "sd_time_s": 7.0,
        "speedup": 1.4,
        "tokens_per_second": 20.0,
        "quality_score": {},
        "decoding": {"mode": "greedy"},
        "n_prompts": 2,
        "n_warmup": 1,
        "n_repeats": 3,
    }


def test_eval_summary_accepts_dict_quality_score_and_optional_engines():
    summary = _summary()
    summary["engines"] = {"hf": {"speedup": 1.4}}
    validate_eval_summary(summary)


def test_eval_summary_rejects_missing_required_field():
    summary = _summary()
    summary.pop("speedup")
    with pytest.raises(ValueError, match="missing required keys"):
        validate_eval_summary(summary)


def test_eval_summary_rejects_non_dict_quality_score():
    summary = _summary()
    summary["quality_score"] = 1.0
    with pytest.raises(ValueError, match="quality_score"):
        validate_eval_summary(summary)

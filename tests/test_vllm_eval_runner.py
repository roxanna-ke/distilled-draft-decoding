from __future__ import annotations

import sys
import types

import pytest

from kdsd.eval.runner import PromptRecord
from kdsd.eval.vllm_runner import run_vllm_eval
from kdsd.utils.io import validate_eval_summary


class TinyTokenizer:
    chat_template = "tiny-chat-template"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<user>{messages[0]['content']}<assistant>"


class FakeSamplingParams:
    last_kwargs = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeSamplingParams.last_kwargs = kwargs


class FakeMetric:
    def __init__(self, name, value=0, values=None):
        self.name = name
        self.value = value
        self.values = [] if values is None else values


class FakeCompletion:
    def __init__(self, text: str, token_ids: list[int]):
        self.text = text
        self.token_ids = token_ids


class FakeRequestOutput:
    def __init__(self, text: str, token_ids: list[int]):
        self.outputs = [FakeCompletion(text, token_ids)]


class FakeLLM:
    init_kwargses = []
    generate_calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeLLM.init_kwargses.append(kwargs)

    def generate(self, prompts, sampling_params, use_tqdm=True):
        FakeLLM.generate_calls.append({
            "prompts": list(prompts),
            "sampling_params": sampling_params,
            "use_tqdm": use_tqdm,
            "speculative": "speculative_config" in self.kwargs,
        })
        return [
            FakeRequestOutput(f" generated-{idx} ", [idx, idx + 1])
            for idx, _ in enumerate(prompts)
        ]

    def get_metrics(self):
        if "speculative_config" not in self.kwargs:
            return []
        return [
            FakeMetric("vllm:spec_decode_num_drafts", value=3),
            FakeMetric("vllm:spec_decode_num_draft_tokens", value=12),
            FakeMetric("vllm:spec_decode_num_accepted_tokens", value=6),
            FakeMetric("vllm:spec_decode_num_accepted_tokens_per_pos", values=[3, 2, 1]),
        ]


@pytest.fixture
def fake_vllm(monkeypatch):
    FakeLLM.init_kwargses = []
    FakeLLM.generate_calls = []
    FakeSamplingParams.last_kwargs = None
    module = types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", module)
    return module


def _runtime():
    return {
        "gamma": 4,
        "max_new_tokens": 8,
        "mode": "greedy",
        "temperature": 0.8,
        "top_p": 0.7,
    }


def _eval_cfg(run_vanilla_baseline=True):
    return {
        "n_warmup": 0,
        "n_repeats": 1,
        "run_vanilla_baseline": run_vanilla_baseline,
        "vllm": {
            "request_batch_size": 8,
            "tensor_parallel_size": 1,
            "draft_tensor_parallel_size": 1,
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.9,
            "swap_space": 0,
            "enforce_eager": False,
            "disable_log_stats": False,
            "max_num_seqs": None,
        },
    }


def test_vllm_eval_uses_draft_model_speculative_config(fake_vllm):
    prompts = [
        PromptRecord(id="p0", prompt_text="first"),
        PromptRecord(id="p1", prompt_text="second"),
    ]

    summary, rows = run_vllm_eval(
        tokenizer=TinyTokenizer(),
        prompts=prompts,
        runtime=_runtime(),
        eval_cfg=_eval_cfg(),
        target_id="target-model",
        draft_id="draft-model",
        run_name="run",
        benchmarks=[],
        dtype="bfloat16",
        trust_remote_code=True,
        seed=7,
    )

    validate_eval_summary(summary)
    spec_kwargs = FakeLLM.init_kwargses[0]
    assert spec_kwargs["model"] == "target-model"
    assert spec_kwargs["tokenizer"] == "target-model"
    assert spec_kwargs["dtype"] == "bfloat16"
    assert spec_kwargs["trust_remote_code"] is True
    assert spec_kwargs["disable_log_stats"] is False
    assert spec_kwargs["speculative_config"] == {
        "method": "draft_model",
        "model": "draft-model",
        "num_speculative_tokens": 4,
        "draft_tensor_parallel_size": 1,
        "max_model_len": 2048,
        "enforce_eager": False,
    }
    assert "speculative_config" not in FakeLLM.init_kwargses[1]
    assert FakeSamplingParams.last_kwargs == {
        "max_tokens": 8,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    assert summary["engines"]["vllm"]["acceptance_rate"] == 0.5
    assert summary["engines"]["vllm"]["avg_accepted_tokens"] == 2.0
    assert summary["engines"]["vllm"]["accepted_tokens_per_pos"] == [3.0, 2.0, 1.0]
    assert summary["acceptance_rate"] == 0.5
    assert summary["avg_accepted_tokens"] == 2.0
    assert rows[0]["generation"] == " generated-0 "
    assert rows[0]["accepted_lens"] == []
    assert rows[0]["n_new_tokens"] == 2
    assert rows[0]["times"]["sd_s_avg"] >= 0.0
    assert rows[0]["times"]["vanilla_s_avg"] is not None


def test_vllm_eval_null_draft_creates_non_speculative_engine(fake_vllm):
    summary, rows = run_vllm_eval(
        tokenizer=TinyTokenizer(),
        prompts=[PromptRecord(id="p0", prompt_text="hello")],
        runtime=_runtime(),
        eval_cfg=_eval_cfg(),
        target_id="target-model",
        draft_id=None,
        run_name="vanilla",
        benchmarks=[],
        dtype="bfloat16",
        trust_remote_code=False,
        seed=None,
    )

    validate_eval_summary(summary)
    assert len(FakeLLM.init_kwargses) == 1
    assert "speculative_config" not in FakeLLM.init_kwargses[0]
    assert summary["draft"] is None
    assert summary["speedup"] == 1.0
    assert summary["engines"]["vllm"]["num_draft_tokens"] == 0
    assert rows[0]["accepted_lens"] == []

from __future__ import annotations

import sys
import types

import pytest

from kdsd.data.target_generate import generate_target_responses_vllm


class TinyTokenizer:
    chat_template = "tiny-chat-template"
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<user>{messages[0]['content']}<assistant>"

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(ch) for ch in text]}

    def decode(self, input_ids, skip_special_tokens=False):
        assert skip_special_tokens is False
        return "".join(chr(idx) for idx in input_ids)


class FakeSamplingParams:
    last_kwargs = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeSamplingParams.last_kwargs = kwargs


class FakeCompletion:
    def __init__(self, text: str):
        self.text = text


class FakeRequestOutput:
    def __init__(self, text: str):
        self.outputs = [FakeCompletion(text)]


class FakeLLM:
    init_kwargs = None
    generate_calls = []

    def __init__(self, **kwargs):
        FakeLLM.init_kwargs = kwargs

    def generate(self, prompts, sampling_params, use_tqdm=True):
        FakeLLM.generate_calls.append(
            {"prompts": list(prompts), "sampling_params": sampling_params, "use_tqdm": use_tqdm}
        )
        return [FakeRequestOutput(f" generated-{idx} ") for idx, _ in enumerate(prompts)]


@pytest.fixture
def fake_vllm(monkeypatch):
    FakeLLM.init_kwargs = None
    FakeLLM.generate_calls = []
    FakeSamplingParams.last_kwargs = None
    module = types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", module)
    return module


def test_vllm_generation_preserves_order_and_metadata(fake_vllm):
    rows = [
        {"id": "a", "prompt_text": "first", "response_text": "old-a", "source": "ultra"},
        {
            "id": "b",
            "prompt_text": "second",
            "response_text": "old-b",
            "source": "ultra",
            "metadata": {"keep": True},
        },
    ]

    out = generate_target_responses_vllm(
        rows,
        model_id="Qwen/test",
        tokenizer=TinyTokenizer(),
        request_batch_size=8,
        max_new_tokens=32,
        mode="greedy",
        seed=7,
        dtype="bfloat16",
        trust_remote_code=False,
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        swap_space=0,
        enforce_eager=False,
        show_progress=False,
    )

    assert [row["id"] for row in out] == ["a", "b"]
    assert [row["response_text"] for row in out] == ["generated-0", "generated-1"]
    assert [row["source"] for row in out] == ["target", "target"]
    assert out[0]["metadata"]["original_response_text"] == "old-a"
    assert out[1]["metadata"]["original_response_text"] == "old-b"
    assert out[1]["metadata"]["keep"] is True
    assert "response_generated_at" in out[0]["metadata"]
    assert FakeLLM.generate_calls[0]["prompts"] == [
        "<user>first<assistant>",
        "<user>second<assistant>",
    ]


def test_vllm_greedy_sampling_params_and_engine_defaults(fake_vllm):
    generate_target_responses_vllm(
        [{"id": "a", "prompt_text": "p", "response_text": "r", "source": "test"}],
        model_id="Qwen/test",
        tokenizer=TinyTokenizer(),
        request_batch_size=1,
        max_new_tokens=64,
        mode="greedy",
        temperature=0.9,
        top_p=0.5,
        seed=11,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        swap_space=0,
        enforce_eager=False,
        show_progress=False,
    )

    assert FakeSamplingParams.last_kwargs == {
        "max_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    assert FakeLLM.init_kwargs == {
        "model": "Qwen/test",
        "tokenizer": "Qwen/test",
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.9,
        "swap_space": 0.0,
        "enforce_eager": False,
        "seed": 11,
    }


def test_vllm_missing_dependency_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)

    with pytest.raises(RuntimeError, match="target_generation.backend=vllm requires vLLM"):
        generate_target_responses_vllm(
            [{"id": "a", "prompt_text": "p", "response_text": "r", "source": "test"}],
            model_id="Qwen/test",
            tokenizer=TinyTokenizer(),
            request_batch_size=1,
            max_new_tokens=16,
            show_progress=False,
        )


def test_vllm_truncates_overlong_prompts_and_marks_metadata(fake_vllm):
    rows = [
        {
            "id": "long",
            "prompt_text": "abcdefghijklmnopqrstuvwxyz",
            "response_text": "old",
            "source": "test",
        }
    ]

    out = generate_target_responses_vllm(
        rows,
        model_id="Qwen/test",
        tokenizer=TinyTokenizer(),
        request_batch_size=1,
        max_new_tokens=4,
        max_model_len=12,
        max_prompt_tokens=11,
        show_progress=False,
    )

    assert FakeLLM.generate_calls[0]["prompts"] == ["<assistant>"]
    assert out[0]["metadata"]["target_prompt_truncated"] is True
    assert out[0]["metadata"]["target_prompt_tokens_original"] > 11
    assert out[0]["metadata"]["target_prompt_tokens_used"] == 11

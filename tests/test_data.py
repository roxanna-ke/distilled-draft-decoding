from pathlib import Path

import torch

from kdsd.data.dataset import KDCollator, KDDataset, format_prompt, tokenize_record
from kdsd.utils.io import write_jsonl


class TinyTokenizer:
    name_or_path = "tiny-tokenizer"
    chat_template = "tiny-chat-template"
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        text = messages[0]["content"]
        return f"<user>{text}<assistant>"

    def __call__(self, text, add_special_tokens=False, **kwargs):
        if isinstance(text, list):
            raise NotImplementedError
        return {"input_ids": [2 + (ord(ch) % 101) for ch in text]}


def test_prompt_masking_sets_labels_and_response_mask():
    tok = TinyTokenizer()
    row = {"prompt_text": "abc", "response_text": "xy"}
    ex = tokenize_record(row, tok, max_seq_len=128)
    prompt_len = len(tok(format_prompt(tok, "abc"))["input_ids"])

    assert ex["labels"][:prompt_len].eq(-100).all()
    assert ex["response_mask"][:prompt_len].logical_not().all()
    assert ex["response_mask"][prompt_len:].all()
    assert ex["labels"][prompt_len:].ne(-100).all()


def test_tokenized_cache_fingerprint_changes_with_policy(tmp_path: Path):
    tok = TinyTokenizer()
    src = tmp_path / "train.jsonl"
    write_jsonl(src, [{"id": "1", "prompt_text": "a", "response_text": "b", "source": "test"}])

    ds_short = KDDataset(src, tok, max_seq_len=64, cache_dir=tmp_path / "cache")
    ds_long = KDDataset(src, tok, max_seq_len=96, cache_dir=tmp_path / "cache")

    assert len(ds_short) == 1
    assert len(ds_long) == 1
    assert ds_short.cache_path != ds_long.cache_path
    assert ds_short.cache_path is not None and ds_short.cache_path.exists()
    assert ds_long.cache_path is not None and ds_long.cache_path.exists()


def test_collator_pads_labels_and_masks(tmp_path: Path):
    tok = TinyTokenizer()
    src = tmp_path / "train.jsonl"
    write_jsonl(
        src,
        [
            {"id": "1", "prompt_text": "a", "response_text": "b", "source": "test"},
            {"id": "2", "prompt_text": "aaaa", "response_text": "bb", "source": "test"},
        ],
    )
    ds = KDDataset(src, tok, max_seq_len=128, cache_dir=None)
    batch = KDCollator(tok)([ds[0], ds[1]])

    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"][0, -1].item() in {-100, batch["input_ids"][0, -1].item()}
    assert batch["response_mask"].dtype is torch.bool

from kdsd.eval import runner
from kdsd.sd.instrument import SDStats


class DummyTokenizer:
    chat_template = None


def test_speedup_reports_wall_time_ratio(monkeypatch):
    def fake_generate_one(*, draft, **kwargs):
        if draft is None:
            stats = SDStats(total_new_tokens=8)
            return "vanilla", stats, 8.0
        stats = SDStats(
            accepted_lens=[2],
            target_calls=1,
            draft_calls=4,
            total_new_tokens=4,
        )
        return "sd", stats, 2.0

    monkeypatch.setattr(runner, "_generate_one", fake_generate_one)
    summary, _ = runner.run_hf_eval(
        target=object(),
        draft=object(),
        tokenizer=DummyTokenizer(),
        prompts=[runner.PromptRecord(id="p0", prompt_text="hello")],
        runtime={
            "gamma": 4,
            "max_new_tokens": 8,
            "mode": "greedy",
            "temperature": 1.0,
            "top_p": 1.0,
        },
        eval_cfg={"n_warmup": 0, "n_repeats": 1, "run_vanilla_baseline": True},
        device="cpu",
        target_id="target",
        draft_id="draft",
        run_name="run",
        benchmarks=[],
    )

    assert summary["vanilla_time_s"] == 8.0
    assert summary["sd_time_s"] == 2.0
    assert summary["speedup"] == 4.0
    assert summary["tokens_per_second"] == 2.0

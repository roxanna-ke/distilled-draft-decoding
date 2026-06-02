model:
  target: Qwen/Qwen2.5-3B-Instruct
  draft_default: Qwen/Qwen2.5-0.5B-Instruct
  dtype: bfloat16
  device: cuda
  attn_impl: sdpa
  trust_remote_code: false
data:
  id: ultrachat_50k
  family: ultrachat
  response_source: original
  n_samples: 50000
  val_samples: 512
  eval_samples: 256
  max_seq_len: 512
  hf_dataset:
    name: HuggingFaceH4/ultrachat_200k
    split: train_sft
  processed_dir: ${data_root}/processed/${data.id}
  target_generated_dir: ${data_root}/target_generated/${data.id}
  tokenized_cache_dir: ${data_root}/tokenized
  train_path: ${data.processed_dir}/train.jsonl
  val_path: ${data.processed_dir}/val.jsonl
  eval_path: ${data.processed_dir}/eval.jsonl
  target_generation:
    source_processed_dir: ${data.processed_dir}
    output_dir: ${data.target_generated_dir}
    splits:
      - train
      - val
    batch_size: 4
    max_new_tokens: 512
    mode: greedy
    temperature: 0.0
    top_p: 1.0
loss:
  kind: fkl
  alpha: 1.0
  temperature: 1.0
train:
  draft_init: ${model.draft_default}
  max_steps: 8000
  num_train_epochs: 1
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-05
  weight_decay: 0.0
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
  logging_steps: 10
  save_steps: 2000
  eval_steps: 2000
  save_total_limit: 4
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
  greater_is_better: false
  save_best_model: true
  bf16: true
  fp16: false
  gradient_checkpointing: true
  dataloader_drop_last: true
  dataloader_num_workers: 2
  remove_unused_columns: false
  report_to_wandb: true
  resume_from_checkpoint: null
  overfit_samples: 0
  compile_target: false
eval:
  n_warmup: 1
  n_repeats: 3
  run_vanilla_baseline: true
  write_generations: true
runtime:
  mode: sampling
  temperature: 1.0
  top_p: 0.9
  gamma: 4
  max_new_tokens: 256
benchmark:
  benchmarks: []
run_name: fkl_ultra50k_bugfix_s8000_seq512_effbs8_a1_effbs8
seed: 42

#!/usr/bin/env python3
"""Train a train-only answer-action LoRA without dev/test data."""

import argparse
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_sft_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]
    if not rows:
        raise ValueError('Empty SFT dataset')
    for row in rows:
        if row.get('split') != 'train':
            raise ValueError('Every SFT row must come from train')
        if [m['role'] for m in row['messages']] != [
                'system', 'user', 'assistant']:
            raise ValueError('Expected system/user/assistant messages')
        target = json.loads(row['messages'][-1]['content'])
        if set(target) != {'answer', 'evidence_ids'}:
            raise ValueError('Invalid answer-action target')
    return rows


def tokenize_row(row, tokenizer, max_seq_length):
    full = tokenizer.apply_chat_template(
        row['messages'], tokenize=True, add_generation_prompt=False)
    prompt = tokenizer.apply_chat_template(
        row['messages'][:-1], tokenize=True, add_generation_prompt=True)
    if full[:len(prompt)] != prompt:
        raise ValueError('Chat prompt is not a prefix of the full sequence')
    if len(full) > max_seq_length:
        raise ValueError(
            f"{row['id']}: {len(full)} tokens exceed {max_seq_length}")
    return {
        'input_ids': full,
        'attention_mask': [1] * len(full),
        'labels': [-100] * len(prompt) + full[len(prompt):],
    }


class FeatureDataset:
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


def collate(features, pad_token_id):
    import torch

    width = max(len(row['input_ids']) for row in features)
    input_ids, attention_mask, labels = [], [], []
    for row in features:
        padding = width - len(row['input_ids'])
        input_ids.append(row['input_ids'] + [pad_token_id] * padding)
        attention_mask.append(row['attention_mask'] + [0] * padding)
        labels.append(row['labels'] + [-100] * padding)
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--max-steps', type=int,
                        help='Smoke mode: train on the longest samples first')
    parser.add_argument('--resume', help='Trainer checkpoint directory')
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max-steps must be positive')

    config = json.loads(args.config.read_text())
    if config.get('training_split') != 'train':
        raise ValueError('LoRA training must use train only')
    output = args.output_dir or resolve(config['output_dir'])
    if (output.exists() and any(output.iterdir())
            and not args.resume):
        raise ValueError('Use a new output directory or --resume')
    output.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this LoRA configuration')

    data_path = resolve(config['train_data'])
    rows = load_sft_rows(data_path)
    base_path = resolve(config['base_model'])
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, local_files_only=True)
    features = [
        tokenize_row(row, tokenizer, config['max_seq_length'])
        for row in rows
    ]
    lengths = [len(row['input_ids']) for row in features]

    # A one-step smoke run exercises the longest memory-heavy examples.
    if args.max_steps is not None:
        count = (args.max_steps
                 * config['gradient_accumulation_steps']
                 * config['per_device_train_batch_size'])
        features = sorted(
            features, key=lambda row: len(row['input_ids']),
            reverse=True)[:count]

    model_config = AutoConfig.from_pretrained(
        base_path, local_files_only=True)
    if hasattr(model_config, 'use_sliding_window'):
        model_config.use_sliding_window = False
    if hasattr(model_config, 'sliding_window'):
        model_config.sliding_window = None

    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
        local_files_only=True,
    )
    model.config.use_cache = False
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        bias='none',
        target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ],
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=config['num_train_epochs'],
        max_steps=args.max_steps if args.max_steps is not None else -1,
        learning_rate=config['learning_rate'],
        per_device_train_batch_size=(
            config['per_device_train_batch_size']),
        gradient_accumulation_steps=(
            config['gradient_accumulation_steps']),
        warmup_steps=config['warmup_steps'],
        weight_decay=config['weight_decay'],
        max_grad_norm=config['max_grad_norm'],
        lr_scheduler_type='cosine',
        optim='adamw_torch_fused',
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=1,
        save_strategy='no' if args.max_steps is not None else 'epoch',
        save_total_limit=3,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=config['seed'],
        data_seed=config['seed'],
    )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=FeatureDataset(features),
        data_collator=lambda batch: collate(batch, pad_id),
    )

    started = time.monotonic()
    result = trainer.train(resume_from_checkpoint=args.resume)
    elapsed = time.monotonic() - started
    adapter = output / 'adapter'
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    trainer.save_state()
    trainer.save_metrics('train', result.metrics)

    weights = list(adapter.glob('*.safetensors'))
    if not weights:
        raise RuntimeError('Adapter weights were not saved')
    record = {
        'config': config,
        'resolved_output_dir': str(output),
        'smoke_max_steps': args.max_steps,
        'training_split': 'train',
        'dev_or_test_used': False,
        'train_data_sha256': hashlib.sha256(
            data_path.read_bytes()).hexdigest(),
        'dataset_samples': len(rows),
        'samples_in_this_run': len(features),
        'sequence_length_mean': float(np.mean(lengths)),
        'sequence_length_max': max(lengths),
        'train_runtime_seconds': elapsed,
        'peak_gpu_memory_gb': round(
            torch.cuda.max_memory_allocated() / 1024**3, 4),
        'adapter_sha256': {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in weights
        },
        'metrics': result.metrics,
    }
    (output / 'training_record.json').write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

"""Deterministic local causal-LM inference with an optional PEFT adapter."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
_CACHE = {}


def parse_local_spec(label):
    if not label.startswith('local:'):
        raise ValueError('Local model labels must start with local:')
    spec = label[len('local:'):]
    base, marker, adapter = spec.partition('@')
    if not base:
        raise ValueError('Local base-model path is empty')
    if marker and not adapter:
        raise ValueError('Local adapter path is empty')
    return base, adapter or None


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path

def clean_local_text(text):
    """Remove one complete Markdown JSON fence, not arbitrary prose."""
    match = re.fullmatch(
        r'```(?:json)?\s*(\{.*\})\s*```',
        text.strip(),
        flags=re.DOTALL,
    )
    return match.group(1) if match else text


def load_local_model(label):
    """Load once per worker; heavy dependencies are imported only on demand."""
    if label in _CACHE:
        return _CACHE[label]

    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    base, adapter = parse_local_spec(label)
    base_path = resolve_path(base)
    if not base_path.is_dir():
        raise ValueError(f'Missing local base model: {base_path}')

    config = AutoConfig.from_pretrained(
        base_path, local_files_only=True)
    # Course prompts are bounded well below the native context window.
    # Full SDPA avoids the unsupported sliding-window SDPA combination.
    if hasattr(config, 'use_sliding_window'):
        config.use_sliding_window = False
    if hasattr(config, 'sliding_window'):
        config.sliding_window = None

    tokenizer = AutoTokenizer.from_pretrained(
        base_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
        local_files_only=True,
    )
    if adapter:
        from peft import PeftModel
        adapter_path = resolve_path(adapter)
        if not adapter_path.is_dir():
            raise ValueError(f'Missing local adapter: {adapter_path}')
        model = PeftModel.from_pretrained(
            model, adapter_path, local_files_only=True)

    if not torch.cuda.is_available():
        raise RuntimeError('Local coursework inference requires CUDA')
    model = model.to('cuda')
    model.eval()
    _CACHE[label] = (tokenizer, model)
    return tokenizer, model


def generate_local(messages, label, max_new_tokens):
    """Return an OpenAI-shaped response for answer.parse_response()."""
    import torch

    tokenizer, model = load_local_model(label)
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_tensors='pt',
    ).to(model.device)
    prompt_tokens = int(encoded['attention_mask'].sum().item())
    prompt_length = encoded['input_ids'].shape[1]

    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0, prompt_length:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    stopped = bool(
        len(generated)
        and int(generated[-1].item()) == tokenizer.eos_token_id
    )
    return {
        'choices': [{
            'finish_reason': 'stop' if stopped else 'length',
            'message': {'content': text},
        }],
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': len(generated),
        },
    }

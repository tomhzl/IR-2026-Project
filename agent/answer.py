"""Optional, fixed-pipeline answer generation from local source excerpts."""

import json
import os
from pathlib import Path
import re
import shlex
import time
from urllib import error, request
from urllib.parse import urlparse

from agent.prompts import build_messages


DEFAULT_API_BASE = 'https://api.deepseek.com'
SILICONFLOW_API_BASE = 'https://api.siliconflow.cn/v1'


def load_api_environment(path=None):
    """Read explicit .env assignments without executing shell or logging values."""
    path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / '.env'
    if not path.is_file():
        return
    allowed = {'IR2026_API_BASE', 'IR2026_API_KEY', 'DEEPSEEK_API_KEY', 'SILICONFLOW_API_KEY',
               'IR2026_VERIFY_THINKING'}
    for line in path.read_text().splitlines():
        match = re.match(r'^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$', line)
        if match is None or match.group(1) not in allowed:
            continue
        parts = shlex.split(match.group(2), comments=True)
        if len(parts) > 1:
            raise ValueError(f'Invalid .env assignment for {match.group(1)}')
        os.environ.setdefault(match.group(1), parts[0] if parts else '')


class ModelResponseError(RuntimeError):
    def __init__(self, message, input_tokens=None, output_tokens=None):
        super().__init__(message)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class ModelAPIError(RuntimeError):
    def __init__(self, status_code, message=None):
        super().__init__(message if message is not None else f'Model API returned HTTP {status_code}')
        self.status_code = status_code


def extract_percentage_answer(question, snippets):
    """Return a directly stated percentage only when its subject matches the question."""
    lowered = question.casefold()
    if not re.search(r'\b(percent(?:age)?|%)\b', lowered):
        return '', []
    if 'decreas' in lowered or 'declin' in lowered:
        verbs = {'decreased', 'declined', 'fell'}
    elif 'increas' in lowered or 'rose' in lowered:
        verbs = {'increased', 'rose'}
    else:
        return '', []
    question_terms = set(re.findall(r'[a-z]{3,}', lowered)) - {
        'what', 'which', 'percentage', 'percent', 'decrease', 'increase',
        'compared', 'according', 'report', 'fiscal', 'year', 'years',
    }
    pattern = re.compile(
        r'((?:[A-Za-z][A-Za-z-]*\s+){1,8})(decreased|declined|fell|increased|rose)'
        r'\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%', re.IGNORECASE,
    )
    matches = []
    for item in snippets:
        for found in pattern.finditer(item['text']):
            if found.group(2).casefold() not in verbs:
                continue
            subject_terms = set(re.findall(r'[a-z]{3,}', found.group(1).casefold()))
            overlap = len(subject_terms & question_terms)
            if overlap >= 3:
                matches.append((overlap, found.group(3) + '%', item['id']))
    if not matches:
        return '', []
    best = max(score for score, _, _ in matches)
    winners = {(value, snippet_id) for score, value, snippet_id in matches if score == best}
    if len({value for value, _ in winners}) != 1:
        return '', []
    value, snippet_id = sorted(winners)[0]
    return value, [snippet_id]


def generate_answer(question, snippets, model, proposed_answer=None, relaxed=False, verification_hint=None, answer_policy='v1'):
    """Generate or independently verify an answer from the same local excerpts."""
    if model.startswith('local:'):
        raise ValueError('Local model inference has been removed; use an API model label')
    base = os.environ.get('IR2026_API_BASE', DEFAULT_API_BASE).rstrip('/')
    messages = build_messages(question, snippets, proposed_answer, relaxed, verification_hint, answer_policy)
    key = os.environ.get('IR2026_API_KEY')
    siliconflow = urlparse(base).hostname in {'api.siliconflow.cn', 'api.siliconflow.com'}
    if siliconflow:
        key = os.environ.get('SILICONFLOW_API_KEY') or key
    if not key and base == DEFAULT_API_BASE:
        key = os.environ.get('DEEPSEEK_API_KEY')
    if not key:
        raise RuntimeError('Set SILICONFLOW_API_KEY for SiliconFlow, or IR2026_API_KEY for the selected API')
    payload = {
        'model': model,
        'temperature': 0,
        'max_tokens': 2400 if proposed_answer is not None and answer_policy=='verified' else 1200,
        'response_format': {'type': 'json_object'},
        'messages': messages,
    }
    if base == DEFAULT_API_BASE:
        # DeepSeek enables thinking by default; its hidden reasoning can exhaust
        # max_tokens before a short JSON answer is produced.
        payload['thinking'] = {'type': 'disabled'}
    if siliconflow:
        verification_thinking = (answer_policy=='verified' and proposed_answer is not None
                                 and os.environ.get('IR2026_VERIFY_THINKING', '0') == '1')
        payload['enable_thinking'] = verification_thinking
        if verification_thinking:
            payload['thinking_budget'] = 1024
        if proposed_answer is not None and answer_policy=='verified':
            from agent.verification import verification_schema
            payload['response_format'] = verification_schema(snippets)
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = request.Request(
        base + '/chat/completions',
        data=body,
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'},
        method='POST',
    )
    for attempt in range(3):
        try:
            with request.urlopen(req, timeout=60) as response:
                data = json.load(response)
            break
        except error.HTTPError as exc:
            if exc.code == 429 and attempt < 2:
                delay = exc.headers.get('Retry-After', '') if exc.headers else ''
                time.sleep(min(10, float(delay)) if delay.isdigit() else 2 ** attempt)
                continue
            raise ModelAPIError(exc.code) from exc
        except error.URLError as exc:
            raise ModelAPIError(None, f'Model API connection failed: {exc.reason}') from exc
        except TimeoutError as exc:
            raise ModelAPIError(None, 'Model API request timed out; token usage is unavailable') from exc
    return parse_response(data)


def parse_response(data):
    """Parse the API's JSON answer and reported token usage."""
    usage = data.get('usage') or {}
    input_tokens = usage.get('prompt_tokens')
    output_tokens = usage.get('completion_tokens')
    if type(input_tokens) is not int or input_tokens < 0:
        input_tokens = None
    if type(output_tokens) is not int or output_tokens < 0:
        output_tokens = None
    choices = data.get('choices') or []
    if not choices or choices[0].get('finish_reason') != 'stop':
        reason = choices[0].get('finish_reason') if choices else 'no_choices'
        raise ModelResponseError(f'Model response was incomplete (finish_reason={reason})',
                                 input_tokens, output_tokens)
    content = choices[0].get('message', {}).get('content')
    if not isinstance(content, str) or not content.strip():
        raise ModelResponseError('Model returned an empty response', input_tokens, output_tokens)
    try:
        parsed = json.loads(content)
    except ValueError as exc:
        raise ModelResponseError('Model returned invalid JSON', input_tokens, output_tokens) from exc
    if not isinstance(parsed, dict):
        raise ModelResponseError('Model did not return a JSON object', input_tokens, output_tokens)
    return parsed, input_tokens, output_tokens

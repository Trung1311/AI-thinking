import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from django.conf import settings

from ..exceptions import AIServiceUnavailable, InvalidAIResponse


# Ten generated quiz questions (prompt, explanation, five hints, five
# choices each) need ~2.5k output tokens; 2048 cut that JSON off mid-way.
MAX_OUTPUT_TOKENS = 8192
MAX_RESPONSE_BYTES = 256_000
GROQ_ENDPOINT = 'https://api.groq.com/openai/v1/chat/completions'

# Cloudflare (fronting Groq's API) bot-flags urllib's default User-Agent
# ("Python-urllib/3.x") with a bare 403 before the request ever reaches
# Groq's own API layer — a generic browser-like one gets through.
_USER_AGENT = 'Mozilla/5.0'


class GroqProvider:
    """OpenAI-compatible Groq adapter — same request/response shape as
    DeepSeekProvider (Groq exposes the same Chat Completions contract),
    just a different endpoint, default model, and the User-Agent header
    Cloudflare requires. Returns untrusted structured data; the existing
    orchestrator/schema validators must validate the returned dictionary.
    """

    def __init__(self, *, api_key=None, model=None, timeout=None, opener=None):
        self.api_key = api_key or os.environ.get('GROQ_API_KEY', '')
        self.model = model or getattr(settings, 'GROQ_MODEL', 'llama-3.3-70b-versatile')
        self.timeout = timeout or getattr(settings, 'GROQ_TIMEOUT_SECONDS', 20)
        self._opener = opener or urlopen

    def generate(self, *, system_prompt, user_prompt, response_schema=None):
        if not self.api_key:
            raise AIServiceUnavailable('Groq is not configured.')

        schema_json = json.dumps(response_schema or {'type': 'object'}, ensure_ascii=False)
        structured_system_prompt = (
            f'{system_prompt.rstrip()}\n\n'
            'Return exactly one valid JSON object and no surrounding markdown or commentary. '
            f'The JSON object must follow this schema: {schema_json}'
        )
        payload = json.dumps({
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': structured_system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            'response_format': {'type': 'json_object'},
            'max_tokens': MAX_OUTPUT_TOKENS,
            'stream': False,
        }).encode('utf-8')
        request = Request(
            GROQ_ENDPOINT,
            data=payload,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'User-Agent': _USER_AGENT,
            },
            method='POST',
        )

        try:
            with self._opener(request, timeout=self.timeout) as response:
                raw_response = response.read(MAX_RESPONSE_BYTES + 1)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise AIServiceUnavailable('The Groq service is unavailable.') from exc

        if len(raw_response) > MAX_RESPONSE_BYTES:
            raise InvalidAIResponse('Groq returned an oversized response.')

        try:
            envelope = json.loads(raw_response)
            choice = envelope['choices'][0]
            finish_reason = choice['finish_reason']
            content = choice['message']['content']
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise InvalidAIResponse('Groq returned an invalid response envelope.') from exc

        if finish_reason != 'stop':
            raise InvalidAIResponse(
                f'Groq did not complete the structured response ({finish_reason}).'
            )
        if not isinstance(content, str) or not content.strip():
            raise InvalidAIResponse('Groq returned no structured response.')
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise InvalidAIResponse('Groq returned invalid JSON.') from exc
        if not isinstance(parsed, dict):
            raise InvalidAIResponse('Groq must return one JSON object.')
        return parsed

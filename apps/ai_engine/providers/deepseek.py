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


class DeepSeekProvider:
    """OpenAI-compatible DeepSeek adapter that returns untrusted structured data.

    Workflow permissions and mastery remain outside this provider. The existing
    orchestrator/schema validators must validate the returned dictionary.
    """

    def __init__(self, *, api_key=None, model=None, base_url=None, timeout=None, opener=None):
        self.api_key = api_key or os.environ.get('DEEPSEEK_API_KEY', '')
        self.model = model or getattr(settings, 'DEEPSEEK_MODEL', 'deepseek-v4-flash')
        configured_base_url = base_url or getattr(
            settings,
            'DEEPSEEK_BASE_URL',
            'https://api.deepseek.com',
        )
        self.execute_url = f'{configured_base_url.rstrip("/")}/chat/completions'
        self.timeout = timeout or getattr(settings, 'DEEPSEEK_TIMEOUT_SECONDS', 20)
        self._opener = opener or urlopen

    def generate(self, *, system_prompt, user_prompt, response_schema=None):
        if not self.api_key:
            raise AIServiceUnavailable('DeepSeek is not configured.')

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
            self.execute_url,
            data=payload,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
            method='POST',
        )

        try:
            with self._opener(request, timeout=self.timeout) as response:
                raw_response = response.read(MAX_RESPONSE_BYTES + 1)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise AIServiceUnavailable('The DeepSeek service is unavailable.') from exc

        if len(raw_response) > MAX_RESPONSE_BYTES:
            raise InvalidAIResponse('DeepSeek returned an oversized response.')

        try:
            envelope = json.loads(raw_response)
            choice = envelope['choices'][0]
            finish_reason = choice['finish_reason']
            content = choice['message']['content']
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise InvalidAIResponse('DeepSeek returned an invalid response envelope.') from exc

        if finish_reason != 'stop':
            raise InvalidAIResponse(
                f'DeepSeek did not complete the structured response ({finish_reason}).'
            )
        if not isinstance(content, str) or not content.strip():
            raise InvalidAIResponse('DeepSeek returned no structured response.')
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise InvalidAIResponse('DeepSeek returned invalid JSON.') from exc
        if not isinstance(parsed, dict):
            raise InvalidAIResponse('DeepSeek must return one JSON object.')
        return parsed

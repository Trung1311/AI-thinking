import json
import os
from copy import deepcopy

from django.conf import settings

from ..exceptions import AIServiceUnavailable, InvalidAIResponse


# Teach-Back evaluates five fields (512 tokens truncated it), and ten
# generated language-quiz questions need ~2.5k tokens (2048 truncated those).
MAX_OUTPUT_TOKENS = 8192


def _gemini_compatible_schema(schema):
    compatible = deepcopy(schema or {})

    def simplify(value):
        if isinstance(value, dict):
            value.pop('additionalProperties', None)
            value.pop('const', None)
            for nested in value.values():
                simplify(nested)
        elif isinstance(value, list):
            for nested in value:
                simplify(nested)

    simplify(compatible)
    return compatible


class GeminiProvider:
    # Gemini accepts inline file parts (PDF, images), so Maths course
    # material can be attached — see client.generate_ai_response(files=...).
    supports_files = True

    def __init__(self, *, api_key=None, model=None, timeout=None, client=None):
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY', '')
        self.model = model or getattr(settings, 'GEMINI_MODEL', 'gemini-flash-latest')
        self.timeout = timeout or getattr(settings, 'GEMINI_TIMEOUT_SECONDS', 20)
        self._client = client

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise AIServiceUnavailable('Gemini is not configured.')
        try:
            from google import genai
        except ImportError as exc:
            raise AIServiceUnavailable('The Google Gen AI SDK is not installed.') from exc
        self._client = genai.Client(
            api_key=self.api_key,
            http_options={'timeout': self.timeout * 1000},
        )
        return self._client

    def generate(self, *, system_prompt, user_prompt, response_schema=None, files=None):
        client = self._get_client()
        from google.genai import errors as genai_errors

        contents = user_prompt
        if files:
            from google.genai import types as genai_types

            contents = [user_prompt] + [
                genai_types.Part.from_bytes(
                    data=file_bytes,
                    mime_type=mime_type or 'application/octet-stream',
                )
                for file_bytes, mime_type in files
            ]

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config={
                    'system_instruction': system_prompt,
                    'response_mime_type': 'application/json',
                    'response_json_schema': _gemini_compatible_schema(response_schema),
                    'max_output_tokens': MAX_OUTPUT_TOKENS,
                    'temperature': 0,
                },
            )
        except genai_errors.APIError as exc:
            # Anything the SDK raises for a failed request (bad/expired
            # model name, rate limit, auth failure, ...) — normalized to
            # AIEngineError so callers trying multiple providers (see
            # apps.ai_engine.providers.fallback.FallbackProvider) can
            # recognize "this provider failed, try the next one" instead
            # of a raw, provider-specific exception type escaping
            # unrecognized and skipping the rest of the chain.
            raise AIServiceUnavailable(f'Gemini request failed: {exc}') from exc
        except OSError as exc:
            raise AIServiceUnavailable('Could not reach Gemini.') from exc
        candidates = getattr(response, 'candidates', None) or []
        finish_reason = getattr(candidates[0], 'finish_reason', None) if candidates else None
        if str(getattr(finish_reason, 'name', finish_reason)) == 'MAX_TOKENS':
            raise InvalidAIResponse('Gemini output was cut off at the token limit.')
        if isinstance(getattr(response, 'parsed', None), dict):
            return response.parsed
        response_text = getattr(response, 'text', '')
        if not response_text:
            raise InvalidAIResponse('Gemini returned no structured response.')
        try:
            return json.loads(response_text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise InvalidAIResponse('Gemini returned invalid JSON.') from exc

import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import SimpleTestCase, override_settings

from .client import generate_ai_response
from .health import probe_ai_provider
from .exceptions import AIServiceUnavailable, InvalidAIResponse
from .orchestrator import (
    orchestrate_diagnostic,
    orchestrate_diagnosis_evaluation,
    orchestrate_teach_back,
)
from .providers.deepseek import DeepSeekProvider
from .providers.fallback import FallbackProvider
from .providers.gemini import GeminiProvider
from .providers.groq import GroqProvider
from .schemas import (
    DiagnosticResponse,
    DiagnosisEvaluationResponse,
    HintResponse,
    TeachBackResponse,
    validate_diagnostic_response,
    validate_diagnosis_evaluation,
    validate_hint_response,
    validate_teach_back_response,
)


VALID_DIAGNOSTIC = {
    'possible_misconception': 'loop-value-misuse',
    'diagnostic_confidence': 0.7,
    'response_type': 'guiding_question',
    'message': 'What value does the loop variable hold?',
    'hint_level': 1,
    'should_reveal_solution': False,
}
TEACH_BACK_FIELDS = ('failure_reason', 'correction', 'concept')
VALID_TEACH_BACK = {
    'field_evaluations': [
        {'field': field, 'understood': True, 'feedback': ''}
        for field in TEACH_BACK_FIELDS
    ],
    'misconception_code': 'none',
    'follow_up_question': '',
}
VALID_DIAGNOSIS_EVALUATION = {
    'understood': False,
    'feedback': 'The answer does not connect the current item to the transformation.',
    'possible_misconception': 'loop-value-misuse',
    'diagnostic_confidence': 0.8,
    'response_type': 'concept_reminder',
    'message': 'Which current item should be transformed during this iteration?',
    'hint_level': 2,
    'should_reveal_solution': False,
}
VALID_HINT = {
    'possible_misconception': 'loop-value-misuse',
    'response_type': 'concept_reminder',
    'message': 'Which current item should be transformed?',
    'hint_level': 2,
    'should_reveal_solution': False,
}


class StaticProvider:
    def __init__(self, payload):
        self.payload = payload

    def generate(self, **kwargs):
        return self.payload


class InvalidThenValidProvider:
    def __init__(self):
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return {**VALID_TEACH_BACK, 'mastery': 'MASTERED'}
        return VALID_TEACH_BACK


class InvalidThenValidDiagnosisProvider:
    def __init__(self):
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return {**VALID_DIAGNOSIS_EVALUATION, 'mastery': 'MASTERED'}
        return VALID_DIAGNOSIS_EVALUATION


class BrokenProvider:
    def generate(self, **kwargs):
        raise RuntimeError('provider failed')


class UnavailableProvider:
    """Raises a recognized AIEngineError, unlike BrokenProvider's bare
    RuntimeError — for exercising FallbackProvider, which only falls
    through to the next provider on this kind of (expected) failure and
    lets anything else propagate as a real bug."""

    def generate(self, **kwargs):
        raise AIServiceUnavailable('provider unavailable')


class AIClientTests(SimpleTestCase):
    @override_settings(AI_PROVIDER_CLASS='')
    def test_missing_provider_fails_safely(self):
        with self.assertRaises(AIServiceUnavailable):
            generate_ai_response(system_prompt='system', user_prompt='learner')

    def test_unexpected_provider_exception_is_normalized(self):
        with self.assertRaises(AIServiceUnavailable):
            generate_ai_response(
                system_prompt='system',
                user_prompt='signals',
                provider=BrokenProvider(),
            )


class DiagnosticSchemaTests(SimpleTestCase):
    def test_valid_response_is_normalized(self):
        response = validate_diagnostic_response(
            VALID_DIAGNOSTIC,
            allowed_misconception_codes=('loop-value-misuse',),
        )
        self.assertIsInstance(response, DiagnosticResponse)
        self.assertFalse(hasattr(response, 'mastery'))

    def test_mastery_or_transition_fields_are_rejected(self):
        for unauthorized in ({'mastery': 'MASTERED'}, {'next_state': 'TRANSFER_TASK'}):
            with self.subTest(unauthorized=unauthorized):
                with self.assertRaises(InvalidAIResponse):
                    validate_diagnostic_response({**VALID_DIAGNOSTIC, **unauthorized})

    def test_solution_reveal_and_excess_hint_level_are_rejected(self):
        revealing = {**VALID_DIAGNOSTIC, 'should_reveal_solution': True}
        excessive = {
            **VALID_DIAGNOSTIC,
            'response_type': 'partial_method',
            'hint_level': 4,
        }
        for payload in (revealing, excessive):
            with self.subTest(payload=payload):
                with self.assertRaises(InvalidAIResponse):
                    validate_diagnostic_response(payload)

    def test_uncurated_misconception_code_is_rejected(self):
        with self.assertRaises(InvalidAIResponse):
            validate_diagnostic_response(
                {**VALID_DIAGNOSTIC, 'possible_misconception': 'invented-code'},
                allowed_misconception_codes=('loop-value-misuse',),
            )

    def test_multiple_questions_and_related_examples_are_rejected(self):
        invalid_messages = (
            'What value changes? When is it appended?',
            'For example, `for item in values`: what is item?',
        )
        for message in invalid_messages:
            with self.subTest(message=message):
                with self.assertRaises(InvalidAIResponse):
                    validate_diagnostic_response({**VALID_DIAGNOSTIC, 'message': message})


class DiagnosticOrchestratorTests(SimpleTestCase):
    def test_valid_provider_response_is_used(self):
        result = orchestrate_diagnostic(
            system_prompt='system',
            user_prompt='privacy-minimized signals',
            curated_fallback=VALID_DIAGNOSTIC,
            allowed_misconception_codes=('loop-value-misuse',),
            provider=StaticProvider(VALID_DIAGNOSTIC),
        )
        self.assertEqual(result.source, 'AI')
        self.assertEqual(result.response.message, VALID_DIAGNOSTIC['message'])

    def test_invalid_or_unavailable_provider_uses_curated_fallback(self):
        invalid = {**VALID_DIAGNOSTIC, 'mastery': 'MASTERED'}
        for provider in (StaticProvider(invalid), BrokenProvider()):
            with self.subTest(provider=type(provider).__name__):
                result = orchestrate_diagnostic(
                    system_prompt='system',
                    user_prompt='privacy-minimized signals',
                    curated_fallback=VALID_DIAGNOSTIC,
                    allowed_misconception_codes=('loop-value-misuse',),
                    provider=provider,
                )
                self.assertEqual(result.source, 'CURATED_FALLBACK')
                self.assertFalse(result.response.should_reveal_solution)
                self.assertTrue(result.failure_code)


class DiagnosisEvaluationSchemaTests(SimpleTestCase):
    options = {
        'allowed_misconception_codes': ('loop-value-misuse',),
        'response_type': 'concept_reminder',
        'hint_level': 2,
        'should_reveal_solution': False,
    }

    def test_valid_evaluation_is_normalized(self):
        response = validate_diagnosis_evaluation(
            VALID_DIAGNOSIS_EVALUATION,
            **self.options,
        )

        self.assertIsInstance(response, DiagnosisEvaluationResponse)
        self.assertFalse(response.understood)

    def test_understood_evaluation_does_not_require_an_unused_next_question(self):
        response = validate_diagnosis_evaluation(
            {
                **VALID_DIAGNOSIS_EVALUATION,
                'understood': True,
                'message': '',
            },
            **self.options,
        )

        self.assertTrue(response.understood)
        self.assertEqual(response.message, '')

    def test_ai_cannot_change_hint_level_reveal_or_workflow(self):
        invalid_payloads = (
            {**VALID_DIAGNOSIS_EVALUATION, 'hint_level': 4},
            {**VALID_DIAGNOSIS_EVALUATION, 'should_reveal_solution': True},
            {**VALID_DIAGNOSIS_EVALUATION, 'next_state': 'GUIDED_REVISION'},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(InvalidAIResponse):
                    validate_diagnosis_evaluation(payload, **self.options)

    def test_invalid_ai_evaluation_uses_curated_fallback(self):
        result = orchestrate_diagnosis_evaluation(
            system_prompt='evaluate semantically',
            user_prompt='privacy-minimized question and answer',
            curated_fallback=VALID_DIAGNOSIS_EVALUATION,
            provider=StaticProvider({**VALID_DIAGNOSIS_EVALUATION, 'mastery': 'MASTERED'}),
            **self.options,
        )

        self.assertEqual(result.source, 'CURATED_FALLBACK')
        self.assertTrue(result.failure_code)

    def test_invalid_structured_response_is_retried_once(self):
        provider = InvalidThenValidDiagnosisProvider()

        result = orchestrate_diagnosis_evaluation(
            system_prompt='evaluate semantically',
            user_prompt='privacy-minimized question and answer',
            curated_fallback=VALID_DIAGNOSIS_EVALUATION,
            provider=provider,
            **self.options,
        )

        self.assertEqual(result.source, 'AI')
        self.assertEqual(provider.calls, 2)


class HintResponseSchemaTests(SimpleTestCase):
    options = {
        'allowed_misconception_codes': ('loop-value-misuse',),
        'response_type': 'concept_reminder',
        'hint_level': 2,
        'should_reveal_solution': False,
    }

    def test_server_selected_hint_is_valid(self):
        response = validate_hint_response(VALID_HINT, **self.options)

        self.assertIsInstance(response, HintResponse)
        self.assertEqual(response.hint_level, 2)

    def test_ai_cannot_skip_level_or_reveal_solution_early(self):
        invalid_payloads = (
            {**VALID_HINT, 'hint_level': 4},
            {**VALID_HINT, 'should_reveal_solution': True},
            {**VALID_HINT, 'message': '```python\ndef double_numbers(values):\n    return values\n```'},
            {**VALID_HINT, 'next_state': 'TEACH_BACK'},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(InvalidAIResponse):
                    validate_hint_response(payload, **self.options)


class TeachBackSchemaTests(SimpleTestCase):
    def test_valid_response_evaluates_every_field_once(self):
        response = validate_teach_back_response(
            VALID_TEACH_BACK,
            expected_fields=TEACH_BACK_FIELDS,
            allowed_misconception_codes=('loop-value-misuse',),
        )

        self.assertIsInstance(response, TeachBackResponse)
        self.assertTrue(all(item.understood for item in response.field_evaluations))

    def test_missing_duplicate_and_unauthorized_fields_are_rejected(self):
        invalid_payloads = (
            {**VALID_TEACH_BACK, 'field_evaluations': VALID_TEACH_BACK['field_evaluations'][:-1]},
            {
                **VALID_TEACH_BACK,
                'field_evaluations': [
                    VALID_TEACH_BACK['field_evaluations'][0],
                    VALID_TEACH_BACK['field_evaluations'][0],
                    VALID_TEACH_BACK['field_evaluations'][2],
                ],
            },
            {**VALID_TEACH_BACK, 'next_state': 'TRANSFER_TASK'},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(InvalidAIResponse):
                    validate_teach_back_response(
                        payload,
                        expected_fields=TEACH_BACK_FIELDS,
                        allowed_misconception_codes=('loop-value-misuse',),
                    )

    def test_revision_feedback_requires_exactly_one_question(self):
        field_evaluations = [dict(item) for item in VALID_TEACH_BACK['field_evaluations']]
        field_evaluations[0] = {
            'field': 'failure_reason',
            'understood': False,
            'feedback': 'The iteration-to-result connection is missing.',
        }
        with self.assertRaises(InvalidAIResponse):
            validate_teach_back_response(
                {**VALID_TEACH_BACK, 'field_evaluations': field_evaluations},
                expected_fields=TEACH_BACK_FIELDS,
                allowed_misconception_codes=('loop-value-misuse',),
            )


class TeachBackOrchestratorTests(SimpleTestCase):
    def test_valid_ai_response_is_used_and_invalid_response_falls_back(self):
        for provider, expected_source in (
            (StaticProvider(VALID_TEACH_BACK), 'AI'),
            (StaticProvider({**VALID_TEACH_BACK, 'mastery': 'MASTERED'}), 'CURATED_FALLBACK'),
        ):
            with self.subTest(expected_source=expected_source):
                result = orchestrate_teach_back(
                    system_prompt='semantic rubric evaluation',
                    user_prompt='privacy-minimized answers',
                    curated_fallback=VALID_TEACH_BACK,
                    expected_fields=TEACH_BACK_FIELDS,
                    allowed_misconception_codes=('loop-value-misuse',),
                    provider=provider,
                )
                self.assertEqual(result.source, expected_source)

    def test_invalid_structured_teach_back_is_retried_once(self):
        provider = InvalidThenValidProvider()

        result = orchestrate_teach_back(
            system_prompt='semantic rubric evaluation',
            user_prompt='privacy-minimized answers',
            curated_fallback=VALID_TEACH_BACK,
            expected_fields=TEACH_BACK_FIELDS,
            allowed_misconception_codes=('loop-value-misuse',),
            provider=provider,
        )

        self.assertEqual(result.source, 'AI')
        self.assertEqual(provider.calls, 2)


class AIProviderHealthTests(SimpleTestCase):
    @override_settings(AI_PROVIDER_CLASS='')
    def test_unconfigured_provider_reports_fallback_without_network(self):
        result = probe_ai_provider()

        self.assertFalse(result['available'])
        self.assertEqual(result['code'], 'NOT_CONFIGURED')

    def test_structured_probe_reports_available(self):
        result = probe_ai_provider(provider=StaticProvider({'status': 'ok'}))

        self.assertTrue(result['available'])
        self.assertEqual(result['code'], 'OK')

    def test_provider_failure_returns_safe_diagnostics(self):
        result = probe_ai_provider(provider=BrokenProvider())

        self.assertFalse(result['available'])
        self.assertEqual(result['code'], 'AIServiceUnavailable')
        self.assertNotIn('key', result['message'].casefold())


class GeminiProviderTests(SimpleTestCase):
    def test_gemini_uses_structured_json_without_schema_controlling_mastery(self):
        models = Mock()
        models.generate_content.return_value = SimpleNamespace(
            parsed=VALID_DIAGNOSTIC,
            text=json.dumps(VALID_DIAGNOSTIC),
        )
        provider = GeminiProvider(
            api_key='test-key',
            model='gemini-test-model',
            client=SimpleNamespace(models=models),
        )

        result = provider.generate(
            system_prompt='system',
            user_prompt='privacy-minimized signals',
            response_schema=DiagnosticResponse.schema_contract(),
        )

        self.assertEqual(result, VALID_DIAGNOSTIC)
        call = models.generate_content.call_args.kwargs
        self.assertEqual(call['model'], 'gemini-test-model')
        self.assertEqual(call['contents'], 'privacy-minimized signals')
        self.assertEqual(call['config']['response_mime_type'], 'application/json')
        self.assertEqual(call['config']['max_output_tokens'], 8192)
        encoded_schema = json.dumps(call['config']['response_json_schema'])
        self.assertNotIn('additionalProperties', encoded_schema)
        self.assertNotIn('const', encoded_schema)

    def test_gemini_attaches_files_as_inline_parts(self):
        models = Mock()
        models.generate_content.return_value = SimpleNamespace(parsed={'ok': True}, text='')
        provider = GeminiProvider(api_key='test-key', client=SimpleNamespace(models=models))

        provider.generate(
            system_prompt='system', user_prompt='read this',
            files=[(b'%PDF-1.4', 'application/pdf')],
        )

        contents = models.generate_content.call_args.kwargs['contents']
        self.assertEqual(contents[0], 'read this')
        self.assertEqual(contents[1].inline_data.mime_type, 'application/pdf')
        self.assertEqual(contents[1].inline_data.data, b'%PDF-1.4')

    def test_gemini_output_cut_off_at_token_limit_is_invalid(self):
        models = Mock()
        models.generate_content.return_value = SimpleNamespace(
            parsed=None,
            text='{"questions": [{"prompt": "unfinished',
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name='MAX_TOKENS'))],
        )
        provider = GeminiProvider(api_key='test-key', client=SimpleNamespace(models=models))

        with self.assertRaisesMessage(InvalidAIResponse, 'token limit'):
            provider.generate(system_prompt='system', user_prompt='learner')

    def test_sdk_api_error_is_normalized_to_aiengineerror(self):
        # Regression: a failed request (e.g. a deprecated/unavailable
        # model name) raised google.genai.errors.APIError straight
        # through, unrecognized by anything expecting AIEngineError — in
        # particular FallbackProvider, which only moves to the next
        # provider on AIEngineError and let the real exception type skip
        # the rest of the chain instead of falling through to it.
        from google.genai import errors as genai_errors

        models = Mock()
        models.generate_content.side_effect = genai_errors.ClientError(
            404, {'error': {'message': 'model not found'}},
        )
        provider = GeminiProvider(
            api_key='test-key',
            model='gemini-test-model',
            client=SimpleNamespace(models=models),
        )

        with self.assertRaises(AIServiceUnavailable):
            provider.generate(system_prompt='system', user_prompt='learner')

    def test_connection_failure_is_normalized_to_aiengineerror(self):
        models = Mock()
        models.generate_content.side_effect = OSError('offline')
        provider = GeminiProvider(
            api_key='test-key',
            model='gemini-test-model',
            client=SimpleNamespace(models=models),
        )

        with self.assertRaises(AIServiceUnavailable):
            provider.generate(system_prompt='system', user_prompt='learner')


class DeepSeekProviderTests(SimpleTestCase):
    class Response:
        def __init__(self, payload):
            self.payload = json.dumps(payload).encode('utf-8')

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            return BytesIO(self.payload).read(size)

    def provider_for(self, envelope):
        opener = Mock(return_value=self.Response(envelope))
        provider = DeepSeekProvider(
            api_key='test-key',
            model='deepseek-test-model',
            base_url='https://deepseek.test',
            timeout=7,
            opener=opener,
        )
        return provider, opener

    def test_deepseek_requests_json_and_returns_parsed_object(self):
        provider, opener = self.provider_for({
            'choices': [{
                'finish_reason': 'stop',
                'message': {'content': json.dumps(VALID_DIAGNOSTIC)},
            }],
        })
        schema = DiagnosticResponse.schema_contract(
            allowed_misconception_codes=('loop-value-misuse',),
        )

        result = provider.generate(
            system_prompt='Ask one focused question.',
            user_prompt='privacy-minimized signals',
            response_schema=schema,
        )

        self.assertEqual(result, VALID_DIAGNOSTIC)
        sent_request = opener.call_args.args[0]
        sent_payload = json.loads(sent_request.data)
        self.assertEqual(sent_request.full_url, 'https://deepseek.test/chat/completions')
        self.assertEqual(sent_payload['model'], 'deepseek-test-model')
        self.assertEqual(sent_payload['response_format'], {'type': 'json_object'})
        self.assertEqual(sent_payload['messages'][1]['content'], 'privacy-minimized signals')
        self.assertIn('valid JSON object', sent_payload['messages'][0]['content'])
        self.assertIn('possible_misconception', sent_payload['messages'][0]['content'])
        self.assertEqual(opener.call_args.kwargs['timeout'], 7)

    def test_empty_malformed_and_truncated_responses_fail_closed(self):
        envelopes = (
            {
                'choices': [{
                    'finish_reason': 'stop',
                    'message': {'content': ''},
                }],
            },
            {
                'choices': [{
                    'finish_reason': 'stop',
                    'message': {'content': '{invalid'},
                }],
            },
            {
                'choices': [{
                    'finish_reason': 'length',
                    'message': {'content': json.dumps(VALID_DIAGNOSTIC)},
                }],
            },
        )
        for envelope in envelopes:
            with self.subTest(envelope=envelope):
                provider, _ = self.provider_for(envelope)
                with self.assertRaises(InvalidAIResponse):
                    provider.generate(
                        system_prompt='Return JSON.',
                        user_prompt='signals',
                        response_schema={},
                    )

    @patch.dict('os.environ', {'DEEPSEEK_API_KEY': ''})
    def test_missing_key_fails_safely(self):
        provider = DeepSeekProvider(api_key='', opener=Mock())
        with self.assertRaises(AIServiceUnavailable):
            provider.generate(
                system_prompt='Return JSON.',
                user_prompt='signals',
                response_schema={},
            )

    def test_transport_failure_is_normalized(self):
        provider = DeepSeekProvider(
            api_key='test-key',
            opener=Mock(side_effect=OSError('offline')),
        )

        with self.assertRaises(AIServiceUnavailable):
            provider.generate(
                system_prompt='Return JSON.',
                user_prompt='signals',
                response_schema={},
            )

    def test_unauthorized_fields_are_rejected_by_existing_schema_validator(self):
        provider, _ = self.provider_for({
            'choices': [{
                'finish_reason': 'stop',
                'message': {'content': json.dumps({
                    **VALID_DIAGNOSTIC,
                    'mastery': 'MASTERED',
                })},
            }],
        })
        payload = provider.generate(
            system_prompt='Return JSON.',
            user_prompt='signals',
            response_schema=DiagnosticResponse.schema_contract(),
        )

        with self.assertRaises(InvalidAIResponse):
            validate_diagnostic_response(payload)


class GroqProviderTests(SimpleTestCase):
    """Same OpenAI-compatible Chat Completions contract as DeepSeek — see
    DeepSeekProviderTests above, which this mirrors closely."""

    class Response:
        def __init__(self, payload):
            self.payload = json.dumps(payload).encode('utf-8')

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            return BytesIO(self.payload).read(size)

    def provider_for(self, envelope):
        opener = Mock(return_value=self.Response(envelope))
        provider = GroqProvider(
            api_key='test-key',
            model='groq-test-model',
            timeout=7,
            opener=opener,
        )
        return provider, opener

    def test_groq_requests_json_and_returns_parsed_object(self):
        provider, opener = self.provider_for({
            'choices': [{
                'finish_reason': 'stop',
                'message': {'content': json.dumps(VALID_DIAGNOSTIC)},
            }],
        })
        schema = DiagnosticResponse.schema_contract(
            allowed_misconception_codes=('loop-value-misuse',),
        )

        result = provider.generate(
            system_prompt='Ask one focused question.',
            user_prompt='privacy-minimized signals',
            response_schema=schema,
        )

        self.assertEqual(result, VALID_DIAGNOSTIC)
        sent_request = opener.call_args.args[0]
        sent_payload = json.loads(sent_request.data)
        self.assertEqual(sent_request.full_url, 'https://api.groq.com/openai/v1/chat/completions')
        self.assertEqual(sent_request.get_header('User-agent'), 'Mozilla/5.0')
        self.assertEqual(sent_payload['model'], 'groq-test-model')
        self.assertEqual(sent_payload['response_format'], {'type': 'json_object'})
        self.assertEqual(sent_payload['messages'][1]['content'], 'privacy-minimized signals')
        self.assertIn('valid JSON object', sent_payload['messages'][0]['content'])
        self.assertEqual(opener.call_args.kwargs['timeout'], 7)

    def test_empty_malformed_and_truncated_responses_fail_closed(self):
        envelopes = (
            {
                'choices': [{
                    'finish_reason': 'stop',
                    'message': {'content': ''},
                }],
            },
            {
                'choices': [{
                    'finish_reason': 'stop',
                    'message': {'content': '{invalid'},
                }],
            },
            {
                'choices': [{
                    'finish_reason': 'length',
                    'message': {'content': json.dumps(VALID_DIAGNOSTIC)},
                }],
            },
        )
        for envelope in envelopes:
            with self.subTest(envelope=envelope):
                provider, _ = self.provider_for(envelope)
                with self.assertRaises(InvalidAIResponse):
                    provider.generate(
                        system_prompt='Return JSON.',
                        user_prompt='signals',
                        response_schema={},
                    )

    @patch.dict('os.environ', {'GROQ_API_KEY': ''})
    def test_missing_key_fails_safely(self):
        provider = GroqProvider(api_key='', opener=Mock())
        with self.assertRaises(AIServiceUnavailable):
            provider.generate(
                system_prompt='Return JSON.',
                user_prompt='signals',
                response_schema={},
            )

    def test_transport_failure_is_normalized(self):
        provider = GroqProvider(
            api_key='test-key',
            opener=Mock(side_effect=OSError('offline')),
        )

        with self.assertRaises(AIServiceUnavailable):
            provider.generate(
                system_prompt='Return JSON.',
                user_prompt='signals',
                response_schema={},
            )


class FallbackProviderTests(SimpleTestCase):
    """FallbackProvider — apps.ai_engine.providers.fallback — tries each
    configured provider in turn, only moving on when one raises an
    AIEngineError (config/transport/invalid-response failure)."""

    def test_first_working_provider_wins_without_touching_the_rest(self):
        first = StaticProvider(VALID_DIAGNOSTIC)
        second = BrokenProvider()
        provider = FallbackProvider(providers=[first, second])

        result = provider.generate(system_prompt='system', user_prompt='learner')

        self.assertEqual(result, VALID_DIAGNOSTIC)

    def test_falls_through_to_the_next_provider_on_failure(self):
        first = UnavailableProvider()
        second = StaticProvider(VALID_DIAGNOSTIC)
        provider = FallbackProvider(providers=[first, second])

        result = provider.generate(system_prompt='system', user_prompt='learner')

        self.assertEqual(result, VALID_DIAGNOSTIC)

    def test_does_not_fall_through_on_an_unrecognized_provider_bug(self):
        # Only AIEngineError (a provider genuinely being unavailable/
        # invalid) triggers fallback — anything else is a real bug and
        # must propagate immediately, not get silently swallowed.
        provider = FallbackProvider(providers=[BrokenProvider(), StaticProvider(VALID_DIAGNOSTIC)])

        with self.assertRaises(RuntimeError):
            provider.generate(system_prompt='system', user_prompt='learner')

    def test_raises_the_last_error_when_every_provider_fails(self):
        provider = FallbackProvider(providers=[UnavailableProvider(), UnavailableProvider()])

        with self.assertRaises(AIServiceUnavailable):
            provider.generate(system_prompt='system', user_prompt='learner')

    def test_no_providers_configured_fails_safely(self):
        provider = FallbackProvider(providers=[])

        with self.assertRaises(AIServiceUnavailable):
            provider.generate(system_prompt='system', user_prompt='learner')

    def test_files_skip_providers_that_cannot_read_them(self):
        # Regression: the shared client refused every file outright after
        # the Maths merge, so uploaded course material never reached AI.
        text_only = Mock(spec=['generate'])
        file_reader = Mock(supports_files=True)
        file_reader.generate.return_value = VALID_DIAGNOSTIC
        provider = FallbackProvider(providers=[text_only, file_reader])
        files = [(b'%PDF-1.4', 'application/pdf')]

        result = generate_ai_response(
            system_prompt='system', user_prompt='learner', files=files, provider=provider,
        )

        self.assertEqual(result, VALID_DIAGNOSTIC)
        text_only.generate.assert_not_called()
        self.assertEqual(file_reader.generate.call_args.kwargs['files'], files)

    def test_files_with_no_file_capable_provider_are_unavailable(self):
        provider = FallbackProvider(providers=[StaticProvider(VALID_DIAGNOSTIC)])

        with self.assertRaises(AIServiceUnavailable):
            generate_ai_response(
                system_prompt='system', user_prompt='learner',
                files=[(b'x', 'image/png')], provider=provider,
            )

    @patch.dict(
        'os.environ',
        {'GEMINI_API_KEY': 'gm-test', 'GROQ_API_KEY': 'gsk-test', 'DEEPSEEK_API_KEY': ''},
        clear=True,
    )
    def test_default_chain_is_built_from_configured_keys_in_priority_order(self):
        provider = FallbackProvider()
        self.assertEqual(
            [type(p).__name__ for p in provider._providers],
            ['GeminiProvider', 'GroqProvider'],
        )

    @override_settings(AI_PROVIDER_CLASS='apps.ai_engine.providers.fallback.FallbackProvider')
    @patch.dict('os.environ', {'GEMINI_API_KEY': '', 'GROQ_API_KEY': 'gsk-test', 'DEEPSEEK_API_KEY': ''})
    def test_generate_ai_response_uses_the_fallback_chain_end_to_end(self):
        with patch(
            'apps.ai_engine.providers.groq.GroqProvider.generate',
            return_value=VALID_DIAGNOSTIC,
        ) as mock_generate:
            result = generate_ai_response(system_prompt='system', user_prompt='learner')

        self.assertEqual(result, VALID_DIAGNOSTIC)
        mock_generate.assert_called_once()

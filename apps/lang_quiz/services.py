from django.core.exceptions import PermissionDenied, ValidationError
from django.db import transaction

from apps.ai_engine.client import generate_ai_response
from apps.ai_engine.exceptions import AIEngineError
from apps.lang_quiz.models import LanguageQuestion
from apps.learning_core.models import (
    Concept,
    HintUsage,
    LearnerAttempt,
    LearningActivity,
    Subject,
    TeachBackAttempt,
    Topic,
    TransferAttempt,
)
from apps.learning_core.services import (
    get_or_create_demo_session,
    mastery_requirements_met,
    record_mastery_decision,
    transition_session,
)
from apps.learning_core.state_machine import WorkflowState, hints_allowed
from apps.learning_core.validators import validate_first_attempt

LANGUAGE_GAP_TYPES = ('vocabulary_gap', 'grammar_misconception', 'context_misunderstanding')

CURATED_HINTS = (
    'Consider which element in the overall sentence or context you should focus on.',
    'Review the basic definition of the relevant vocabulary or grammar rule.',
    'Compare your answer to how the same word or structure is used in similar example sentences.',
    'Check the syntax or keyword in the prompt that gives you a clue to the answer.',
)


@transaction.atomic
def ensure_demo_question():
    """Ensure standard demo subject, topic, concept, activity, and language question exist."""
    subject, _ = Subject.objects.get_or_create(
        slug='languages', defaults={'name': 'Languages', 'description': 'Language Learning'}
    )
    topic, _ = Topic.objects.get_or_create(
        subject=subject,
        slug='english-vocab',
        defaults={'name': 'English Vocabulary', 'description': 'Vocabulary and Context Usage'},
    )
    concept, _ = Concept.objects.get_or_create(
        topic=topic,
        slug='word-meanings',
        defaults={'name': 'Word Meanings', 'description': 'Understand words in appropriate contexts.'},
    )
    activity, _ = LearningActivity.objects.get_or_create(
        concept=concept,
        title='Choose the correct vocabulary in context',
        defaults={
            'activity_type': 'language',
            'prompt': 'Fill in the blank with the appropriate word: "She decided to ____ the offer after careful consideration."',
            'reference_answer': 'accept',
            'rubric': {'target_gap': 'vocabulary_gap', 'requires_transfer': True},
        },
    )
    question, _ = LanguageQuestion.objects.get_or_create(
        activity=activity,
        defaults={
            'prompt': activity.prompt,
            'question_type': LanguageQuestion.QuestionType.VOCABULARY,
            'reference_answer': 'accept',
            'rubric': activity.rubric,
            'transfer_prompt': 'Choose the correct word: "He refused to ____ responsibility for the error."',
            'active': True,
        },
    )
    return question


@transaction.atomic
def create_questions_from_file_upload(file_obj):
    """Extract text from an uploaded file and generate 10 reading + 10 grammar LanguageQuestion instances.

    Uses pypdf for PDF files to extract clean text.
    Reading section: 10 AI-generated reading comprehension questions in English.
    Grammar section: 10 AI-generated grammar questions in English.
    Each section has a Japanese heading stored in title_ja (リーディング質問 / 文法質問).
    """
    import io as _io
    from apps.ai_engine.client import generate_questions_from_text

    filename = file_obj.name.lower()
    raw_text = ''
    try:
        content_bytes = file_obj.read()
        if filename.endswith('.pdf'):
            # Use pypdf to properly extract text from PDF binary
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise ValueError(
                    'PDF support is not installed. Run "pip install -r requirements.txt" '
                    'in the active virtual environment and restart the server.'
                ) from exc
            try:
                reader = PdfReader(_io.BytesIO(content_bytes))
                pages = []
                for page in reader.pages:
                    page_text = page.extract_text() or ''
                    if page_text.strip():
                        pages.append(page_text)
                raw_text = '\n'.join(pages)
            except Exception as exc:
                raise ValueError(
                    'The PDF could not be read. It may be damaged, encrypted, or unsupported.'
                ) from exc
            if not raw_text.strip():
                raise ValueError(
                    'The PDF appears to be a scanned image with no extractable text. '
                    'Please upload a text-based PDF, TXT, or Markdown file.'
                )
        else:
            for enc in ('utf-8-sig', 'utf-8', 'utf-16', 'shift-jis'):
                try:
                    raw_text = content_bytes.decode(enc)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            else:
                raw_text = content_bytes.decode('latin-1', errors='ignore')
    except ValueError:
        raise
    except Exception:
        raw_text = ''

    if not raw_text.strip():
        return []

    # Generate 10 reading comprehension questions and 10 grammar questions separately
    reading_specs = generate_questions_from_text(raw_text, count=10, section='reading')
    grammar_specs = generate_questions_from_text(raw_text, count=10, section='grammar')

    # Tag each spec with its section type and Japanese heading
    for spec in reading_specs:
        spec['question_type'] = LanguageQuestion.QuestionType.READING
        spec['title_ja'] = 'リーディング質問'  # Japanese: "Reading Questions"

    for spec in grammar_specs:
        spec['question_type'] = LanguageQuestion.QuestionType.GRAMMAR
        spec['title_ja'] = '文法質問'  # Japanese: "Grammar Questions"

    all_specs = reading_specs + grammar_specs
    if not all_specs:
        return []

    subject, _ = Subject.objects.get_or_create(
        slug='languages', defaults={'name': 'Languages', 'description': 'Language Learning'}
    )
    topic, _ = Topic.objects.get_or_create(
        subject=subject,
        slug='uploaded-file-quiz',
        defaults={'name': 'Uploaded File Questions', 'description': 'Questions generated from uploaded learning material'},
    )
    concept, _ = Concept.objects.get_or_create(
        topic=topic,
        slug='file-concepts',
        defaults={'name': 'File-based Concepts', 'description': 'AI-generated questions from uploaded files'},
    )

    created_questions = []
    for spec in all_specs:
        activity = LearningActivity.objects.create(
            concept=concept,
            title=f'File Quiz: {spec["reference_answer"]}',
            activity_type='language',
            prompt=spec['prompt'],
            reference_answer=spec['reference_answer'],
            rubric=spec.get('rubric', {}),
        )
        question = LanguageQuestion.objects.create(
            activity=activity,
            prompt=spec['prompt'],
            question_type=spec['question_type'],
            title_ja=spec['title_ja'],
            reference_answer=spec['reference_answer'],
            rubric=spec.get('rubric', {}),
            transfer_prompt=spec.get('transfer_prompt', ''),
            active=True,
        )
        created_questions.append(question)

    return created_questions


def get_demo_session(*, browser_session_key, question):
    """Retrieve or initialize a demo learning session for the question topic."""
    return get_or_create_demo_session(
        browser_session_key=browser_session_key,
        topic=question.activity.concept.topic,
    )


def _fallback_classify_error(*, answer, reasoning, question_type=None, reference_answer=None, rubric=None):
    """Rule-based fallback classification when AI engine is unavailable."""
    answer_str = (answer or '').strip().lower()
    reasoning_str = (reasoning or '').strip().lower()
    ref_str = (reference_answer or '').strip().lower()

    if ref_str and answer_str == ref_str:
        return {
            'gap_type': None,
            'is_correct': True,
            'confidence': 1.0,
            'explanation': 'The learner provided the exact expected answer.',
        }

    default_gap = None
    if rubric and isinstance(rubric, dict) and rubric.get('target_gap') in LANGUAGE_GAP_TYPES:
        default_gap = rubric['target_gap']
    elif question_type == LanguageQuestion.QuestionType.VOCABULARY:
        default_gap = 'vocabulary_gap'
    elif question_type == LanguageQuestion.QuestionType.GRAMMAR:
        default_gap = 'grammar_misconception'
    elif question_type in (LanguageQuestion.QuestionType.READING, LanguageQuestion.QuestionType.WRITTEN):
        default_gap = 'context_misunderstanding'

    combined_text = f'{answer_str} {reasoning_str}'

    if 'vocab' in combined_text or 'word' in combined_text or 'meaning' in combined_text or '単語' in combined_text or '意味' in combined_text:
        gap_type = 'vocabulary_gap'
    elif 'grammar' in combined_text or 'tense' in combined_text or 'verb' in combined_text or 'syntax' in combined_text or '文法' in combined_text:
        gap_type = 'grammar_misconception'
    elif 'context' in combined_text or 'passage' in combined_text or 'read' in combined_text or '文脈' in combined_text:
        gap_type = 'context_misunderstanding'
    else:
        gap_type = default_gap or 'vocabulary_gap'

    return {
        'gap_type': gap_type,
        'is_correct': False,
        'confidence': 0.7,
        'explanation': f'Fallback classification identified {gap_type} based on question type and input heuristics.',
    }


def classify_language_error(*, answer, reasoning, question_type=None, reference_answer=None, rubric=None):
    """Classify a learner's language mistake into vocabulary_gap, grammar_misconception, or context_misunderstanding.

    First attempts structured AI classification if available; gracefully falls back to deterministic rules.
    """
    system_prompt = (
        'You are an AI diagnostic assistant for language learning. '
        'Classify the learner mistake into one of: vocabulary_gap, grammar_misconception, context_misunderstanding. '
        'If the answer is fully correct, set gap_type to "none" and is_correct to true.'
    )
    user_prompt = (
        f'Question Type: {question_type or "General"}\n'
        f'Reference Answer: {reference_answer or "N/A"}\n'
        f'Rubric: {rubric or {}}\n'
        f'Learner Answer: {answer}\n'
        f'Learner Reasoning: {reasoning}'
    )

    try:
        response = generate_ai_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            # Must be real JSON Schema — Gemini answers a bare
            # {'field': 'type-name'} map with a 400 on every call.
            response_schema={
                'type': 'object',
                'properties': {
                    'gap_type': {'type': 'string', 'enum': [*LANGUAGE_GAP_TYPES, 'none']},
                    'is_correct': {'type': 'boolean'},
                    'confidence': {'type': 'number'},
                    'explanation': {'type': 'string'},
                },
                'required': ['gap_type', 'is_correct', 'confidence', 'explanation'],
            },
        )
        if isinstance(response, dict):
            gap_type = response.get('gap_type')
            if gap_type not in LANGUAGE_GAP_TYPES:
                gap_type = None
            return {
                'gap_type': gap_type,
                'is_correct': bool(response.get('is_correct', False)),
                'confidence': float(response.get('confidence', 0.0)),
                'explanation': str(response.get('explanation', '')),
            }
    except AIEngineError:
        pass

    return _fallback_classify_error(
        answer=answer,
        reasoning=reasoning,
        question_type=question_type,
        reference_answer=reference_answer,
        rubric=rubric,
    )


def record_language_attempt_evaluation(*, learner_attempt, question_type=None, reference_answer=None, rubric=None):
    """Classify the error for a learner attempt and store the result in learner_attempt.evaluation."""
    evaluation = classify_language_error(
        answer=learner_attempt.answer,
        reasoning=learner_attempt.reasoning,
        question_type=question_type,
        reference_answer=reference_answer,
        rubric=rubric,
    )
    learner_attempt.evaluation = evaluation
    learner_attempt.save(update_fields=('evaluation',))
    return evaluation


@transaction.atomic
def submit_first_attempt(*, learning_session, question, answer, reasoning, confidence):
    """Process first attempt submission enforcing Think-First Gate (answer, reasoning, confidence)."""
    validate_first_attempt(answer=answer, reasoning=reasoning, confidence=confidence)
    if learning_session.current_state != WorkflowState.DIAGNOSTIC_QUIZ:
        raise ValidationError('A first attempt is accepted only after the diagnostic stage begins.')

    attempt = LearnerAttempt.objects.create(
        learning_session=learning_session,
        activity=question.activity,
        answer=answer,
        reasoning=reasoning,
        confidence=confidence,
    )
    transition_session(learning_session, WorkflowState.FIRST_ATTEMPT)
    transition_session(learning_session, WorkflowState.RESPONSE_EVALUATION)

    evaluation = record_language_attempt_evaluation(
        learner_attempt=attempt,
        question_type=question.question_type,
        reference_answer=question.reference_answer,
        rubric=question.rubric,
    )

    if evaluation.get('is_correct'):
        transition_session(learning_session, WorkflowState.TEACH_BACK)
    else:
        transition_session(learning_session, WorkflowState.DIAGNOSIS)
        transition_session(learning_session, WorkflowState.GUIDED_REVISION)

    return attempt, evaluation


@transaction.atomic
def request_curated_hint(*, learning_session):
    """Unlock next hint level (1 to 4) during guided revision."""
    if not hints_allowed(learning_session.current_state):
        raise PermissionDenied('Hints are available only during guided revision.')
    latest_attempt = learning_session.attempts.order_by('-created_at').first()
    if latest_attempt is None:
        raise PermissionDenied('Submit an attempt before requesting a hint.')

    latest_hint = HintUsage.objects.filter(
        learner_attempt__learning_session=learning_session
    ).order_by('-created_at').first()
    if latest_hint and not learning_session.attempts.filter(
        pk__gt=latest_hint.learner_attempt_id
    ).exists():
        raise PermissionDenied('Revise your work before unlocking the next hint.')

    next_level = 1 if latest_hint is None else latest_hint.level + 1
    if next_level > len(CURATED_HINTS):
        raise PermissionDenied('The four-level hint ladder is complete.')

    return HintUsage.objects.create(
        learner_attempt=latest_attempt,
        level=next_level,
        content=CURATED_HINTS[next_level - 1],
    )


@transaction.atomic
def submit_revised_attempt(*, learning_session, question, answer, reasoning, confidence):
    """Process revised attempt submission after hint usage."""
    validate_first_attempt(answer=answer, reasoning=reasoning, confidence=confidence)
    if learning_session.current_state != WorkflowState.GUIDED_REVISION:
        raise ValidationError('Revised attempt is accepted only during guided revision.')

    latest_attempt = learning_session.attempts.order_by('-created_at').first()
    revision_num = (latest_attempt.revision_number + 1) if latest_attempt else 1

    attempt = LearnerAttempt.objects.create(
        learning_session=learning_session,
        activity=question.activity,
        answer=answer,
        reasoning=reasoning,
        confidence=confidence,
        revision_number=revision_num,
    )

    transition_session(learning_session, WorkflowState.RESPONSE_EVALUATION)
    evaluation = record_language_attempt_evaluation(
        learner_attempt=attempt,
        question_type=question.question_type,
        reference_answer=question.reference_answer,
        rubric=question.rubric,
    )

    if evaluation.get('is_correct'):
        transition_session(learning_session, WorkflowState.TEACH_BACK)
    else:
        transition_session(learning_session, WorkflowState.DIAGNOSIS)
        transition_session(learning_session, WorkflowState.GUIDED_REVISION)

    return attempt, evaluation


def begin_teach_back(*, learning_session, original_passed):
    """Transition to Teach-Back stage when original exercise passed."""
    if not original_passed:
        raise PermissionDenied('Teach-Back requires the original question to be answered correctly.')
    return transition_session(learning_session, WorkflowState.TEACH_BACK)


@transaction.atomic
def submit_teach_back(*, learning_session, response):
    """Submit and evaluate learner Teach-Back explanation."""
    if learning_session.current_state != WorkflowState.TEACH_BACK:
        raise PermissionDenied('Teach-Back responses are accepted only during Teach-Back stage.')
    if not (response or '').strip():
        raise ValidationError('Teach-Back response cannot be empty.')

    evaluation = 'CLEAR_UNDERSTANDING' if len(response.strip()) >= 10 else 'PARTIAL_UNDERSTANDING'
    attempt = TeachBackAttempt.objects.create(
        learning_session=learning_session,
        response=response,
        evaluation=evaluation,
        feedback='Clear understanding demonstrated.' if evaluation == 'CLEAR_UNDERSTANDING' else 'Provide more detail in explanation.',
    )
    return attempt


def begin_transfer_check(*, learning_session, teach_back_evaluation):
    """Transition to Transfer Task stage when Teach-Back is clear."""
    if teach_back_evaluation != 'CLEAR_UNDERSTANDING':
        raise PermissionDenied('Transfer Check requires a clear Teach-Back explanation.')
    return transition_session(learning_session, WorkflowState.TRANSFER_TASK)


@transaction.atomic
def submit_transfer_attempt(*, learning_session, question, response, reasoning, confidence, passed=False):
    """Submit unassisted transfer task attempt."""
    if learning_session.current_state != WorkflowState.TRANSFER_TASK:
        raise PermissionDenied('Transfer attempts are allowed only during Transfer Task stage.')
    validate_first_attempt(answer=response, reasoning=reasoning, confidence=confidence)

    attempt = TransferAttempt.objects.create(
        learning_session=learning_session,
        activity=question.activity,
        response=response,
        reasoning=reasoning,
        confidence=confidence,
        used_assistance=False,
        passed=passed,
    )
    return attempt


def complete_transfer_check(*, learning_session, original_passed, teach_back_clear,
                            transfer_passed, used_assistance, misconception_repeated):
    """Finalize learning state based on transfer result."""
    mastered = mastery_requirements_met(
        original_passed=original_passed,
        teach_back_clear=teach_back_clear,
        transfer_passed=transfer_passed,
        transfer_unassisted=not used_assistance,
        misconception_repeated=misconception_repeated,
    )
    concept = learning_session.topic.concepts.order_by('pk').first()
    if concept is None:
        raise ValidationError('A concept is required before completing the Transfer Check.')
    record = record_mastery_decision(
        learning_session=learning_session,
        concept=concept,
        original_passed=original_passed,
        teach_back_clear=teach_back_clear,
        transfer_passed=transfer_passed,
        transfer_unassisted=not used_assistance,
        repeated_misconception_codes=(('language_misconception',) if misconception_repeated else ()),
        evidence={
            'source': 'language_transfer_check',
            'used_assistance': used_assistance,
        },
    )
    target = WorkflowState(record.status)
    if mastered != (target == WorkflowState.MASTERED):
        raise ValidationError('Inconsistent language mastery evaluation.')
    transition_session(learning_session, target)
    return target

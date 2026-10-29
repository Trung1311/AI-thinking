"""Ten-question language quiz generation and anonymous progress tracking."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re

from django.db import transaction
from django.core.files.base import ContentFile

from apps.ai_engine.client import generate_ai_response
from apps.ai_engine.exceptions import AIEngineError

from .models import LanguageCourseProgress, LanguageQuizRun, MissingLanguageQuestion


SECTION_LABELS = {
    'vocabulary': 'Vocabulary',
    'reading': 'Reading',
    'grammar': 'Grammar',
    'myself': 'Myself',
    'missing': 'Missing',
    'diagnostic': '診断テスト',
}

# A real JSON Schema, wrapped in a top-level object: Gemini rejects anything
# that is not JSON Schema with a 400, and the DeepSeek/Groq adapters use
# JSON-object mode, which can only return an object, never a bare array.
AI_QUESTION_SCHEMA = {
    'type': 'object',
    'properties': {
        'questions': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'prompt': {'type': 'string'},
                    'answer': {'type': 'string'},
                    'skill_focus': {'type': 'string'},
                    'explanation': {'type': 'string'},
                    'next_step': {'type': 'string'},
                    'hints': {'type': 'array', 'items': {'type': 'string'}},
                    'choices': {'type': 'array', 'items': {'type': 'string'}},
                },
                'required': ['prompt', 'answer', 'explanation', 'hints', 'choices'],
            },
        },
    },
    'required': ['questions'],
}


VOCABULARY_ANSWER_MODES = {'multiple_choice', 'typing'}
DIFFICULTY_LABELS = {'beginner', 'intermediate', 'advanced'}
DIFFICULTY_GUIDANCE = {
    'beginner': (
        'Use common everyday words, short sentences, direct clues, basic grammar, '
        'and literal reading comprehension comparable to CEFR A1-A2.'
    ),
    'intermediate': (
        'Use moderately varied vocabulary, multi-clause sentences, common academic contexts, '
        'and a mix of direct and inferential reasoning comparable to CEFR B1-B2.'
    ),
    'advanced': (
        'Use nuanced academic or professional vocabulary, complex grammar and sentence structures, '
        'subtle contextual distinctions, and deeper inference comparable to CEFR C1-C2.'
    ),
}


def _vocabulary_choices(answer, provided=None):
    answer = str(answer).strip()
    candidates = [str(item).strip() for item in (provided or []) if str(item).strip()]
    candidates.extend(word for word, _meaning, _prompt in VOCABULARY_DATA)
    unique = []
    for candidate in [answer, *candidates]:
        if candidate.casefold() not in {item.casefold() for item in unique}:
            unique.append(candidate)
    choices = [answer, *random.sample(unique[1:], min(4, len(unique) - 1))]
    if len(choices) != 5:
        return []
    random.shuffle(choices)
    return choices


def _reading_choices(answer, provided=None):
    """Return exactly five distinct reading answers, including the correct one."""
    answer = str(answer).strip()
    fallback_distractors = {
        'walked': ['took the bus', 'rode a bicycle', 'went by car', 'ran'],
        'it was closed': ['it was full', 'he forgot his card', 'he went to the wrong building', 'it opened at seven'],
        'rain': ['snow', 'sunshine', 'strong wind', 'fog'],
        'blue whale': ['elephant', 'giraffe', 'great white shark', 'hippopotamus'],
        'studying': ['luck', 'exercise', 'sleeping', 'asking a friend'],
        'sunlight': ['moonlight', 'wind', 'soil', 'rainwater'],
        'yes': ['no', 'only outside', 'only with a reservation', 'not after closing'],
        'the baby was sleeping': ['she was angry', 'she had a secret', 'the room was empty', 'she had lost her voice'],
        'flooding': ['roadwork', 'a traffic accident', 'heavy traffic', 'a broken bridge'],
        'two': ['one', 'three', 'four', 'eight'],
        'no': ['yes', 'the task was cancelled', 'someone else finished it', 'Rui did not begin'],
        'free': ['$5', '$10', 'half price', 'the regular weekday price'],
    }
    supplied = [str(item).strip() for item in (provided or []) if str(item).strip()]
    candidates = [
        *supplied,
        *fallback_distractors.get(answer.casefold(), []),
        *(str(item[2]).strip() for item in READING_DATA),
    ]
    distractors = []
    seen = {answer.casefold()}
    for candidate in candidates:
        normalized = candidate.casefold()
        if normalized not in seen:
            distractors.append(candidate)
            seen.add(normalized)
        if len(distractors) == 4:
            break
    if len(distractors) != 4:
        return []
    choices = [answer, *distractors]
    random.shuffle(choices)
    return choices


def _grammar_choices(answer, provided=None):
    """Return exactly five distinct grammar options, including the correct one."""
    answer = str(answer).strip()
    fallback_distractors = {
        'is': ['are', 'am', 'be', 'was'],
        'goes': ['go', 'went', 'going', 'is go'],
        'had': ['have', 'has', 'would have', 'have had'],
        'is read': ['reads', 'read', 'was reading', 'is reading'],
        'than': ['then', 'as', 'from', 'to'],
        'are': ['is', 'am', 'be', 'was'],
        'listening to': ['listen', 'listened to', 'to listen', 'listening'],
        'whether': ['what', 'that', 'which', 'why'],
        'will have completed': ['completed', 'will complete', 'have completed', 'are completing'],
        'who': ['which', 'whose', 'whom', 'where'],
        'could': ['can', 'will', 'should', 'may'],
    }
    supplied = [str(item).strip() for item in (provided or []) if str(item).strip()]
    candidates = [
        *supplied,
        *fallback_distractors.get(answer.casefold(), []),
        *(str(item[2]).strip() for item in GRAMMAR_DATA),
    ]
    distractors = []
    seen = {answer.casefold()}
    for candidate in candidates:
        normalized = candidate.casefold()
        if normalized not in seen:
            distractors.append(candidate)
            seen.add(normalized)
        if len(distractors) == 4:
            break
    if len(distractors) != 4:
        return []
    choices = [answer, *distractors]
    random.shuffle(choices)
    return choices


def _question(
    key, prompt, answer, explanation, next_step, *, section, hints=None,
    answer_mode='typing', choices=None, difficulty='intermediate',
):
    default_hints = [
        'Read the whole question and identify the exact information the blank requires.',
        f'Focus on the {SECTION_LABELS.get(section, section)} rule, meaning, or evidence being tested.',
        f'The answer has {len(str(answer))} characters; narrow it by its grammatical role and context.',
        f'The answer starts with "{str(answer)[:1]}" and ends with "{str(answer)[-1:]}".',
        f'The answer is "{answer}". If you cannot type it, select "Give up".',
    ]
    hints = hints if isinstance(hints, list) and len(hints) >= 5 else default_hints
    supplied_choices = (
        [str(choice).strip() for choice in choices if str(choice).strip()]
        if isinstance(choices, list) else []
    )
    supplied_normalized = {choice.casefold() for choice in supplied_choices}
    has_valid_five_choices = (
        answer_mode == 'multiple_choice'
        and len(supplied_choices) == 5
        and len(supplied_normalized) == 5
        and str(answer).strip().casefold() in supplied_normalized
    )
    if has_valid_five_choices:
        # Preserve authored/PDF choice wording and order exactly as supplied.
        choices = supplied_choices
    elif section == 'vocabulary' and answer_mode == 'multiple_choice':
        choices = _vocabulary_choices(answer, choices)
    elif section == 'reading' and answer_mode == 'multiple_choice':
        choices = _reading_choices(answer, choices)
    elif section == 'grammar' and answer_mode == 'multiple_choice':
        choices = _grammar_choices(answer, choices)
    elif supplied_choices:
        choices = supplied_choices
    else:
        choices = []
    return {
        'key': key,
        'prompt': prompt,
        'answer': str(answer),
        'explanation': explanation,
        'next_step': next_step,
        'hints': list(hints)[:5],
        'section': section,
        'choices': choices,
        'answer_mode': answer_mode,
        'difficulty': difficulty,
        'attempt_count': 0,
        'hint_level': 0,
        'resolved': False,
        'is_correct': None,
        'last_feedback': '',
    }


VOCABULARY_DATA = [
    ('abundant', '豊富な', 'The region has ____ natural resources.'),
    ('accurate', '正確な', 'Please provide ____ information.'),
    ('adapt', '適応する', 'Animals must ____ to environmental change.'),
    ('benefit', '利益・恩恵', 'Exercise has a positive ____ for health.'),
    ('concise', '簡潔な', 'Keep your summary clear and ____.'),
    ('decline', '減少する・断る', 'Sales began to ____ in winter.'),
    ('essential', '不可欠な', 'Water is ____ for life.'),
    ('fragile', '壊れやすい', 'Handle the ____ glass carefully.'),
    ('generate', '生み出す', 'Solar panels ____ electricity.'),
    ('maintain', '維持する', 'It is important to ____ a steady pace.'),
    ('persuade', '説得する', 'She tried to ____ him to join.'),
    ('reliable', '信頼できる', 'We need a ____ source of information.'),
    ('significant', '重要な・かなりの', 'The study found a ____ difference.'),
    ('temporary', '一時的な', 'This is only a ____ solution.'),
    ('verify', '確かめる', 'Always ____ a claim before sharing it.'),
]

GRAMMAR_DATA = [
    ('g1', 'She ____ to school every day. (go)', 'goes', 'Third-person singular present tense requires "goes".'),
    ('g2', 'I have lived here ____ 2020.', 'since', 'Use "since" to indicate the starting point of an action.'),
    ('g3', 'If I ____ more time, I would travel.', 'had', 'Subjunctive past: the if-clause uses the past tense form.'),
    ('g4', 'This book ____ by many students.', 'is read', 'Passive voice = be verb + past participle.'),
    ('g5', 'He is taller ____ his brother.', 'than', 'Comparative adjectives are followed by "than".'),
    ('g6', 'There ____ two apples on the table.', 'are', 'The plural noun "two apples" requires the plural verb "are".'),
    ('g7', 'I enjoy ____ music. (listen)', 'listening to', '"enjoy" is followed by a gerund; "listen" requires "to".'),
    ('g8', 'She asked me ____ I was ready.', 'whether', '"whether" introduces an indirect yes/no question.'),
    ('g9', 'By next year, they ____ the project.', 'will have completed', 'Future perfect expresses an action completed before a future point.'),
    ('g10', 'The man ____ lives next door is a doctor.', 'who', '"who" is the subject relative pronoun for people.'),
    ('g11', 'Neither answer ____ correct.', 'is', '"Neither" is treated as singular and takes a singular verb.'),
    ('g12', 'I wish I ____ speak French.', 'could', 'Use "could" (past form) to express an unattainable present wish.'),
]

READING_DATA = [
    ('r1', 'Mina missed the bus, so she walked to school. How did Mina get to school?', 'walked', 'The text explicitly states she walked because she missed the bus.'),
    ('r2', 'The library closes at six, but Ken arrived at six thirty. Why could he not enter?', 'it was closed', 'Ken arrived 30 minutes after closing time, so the library was shut.'),
    ('r3', 'Aya took an umbrella because dark clouds covered the sky. What weather did she expect?', 'rain', 'Dark clouds and an umbrella signal she expected rain.'),
    ('r4', 'The blue whale is larger than any other animal. Which animal is the largest?', 'blue whale', 'The comparative phrase "larger than any other animal" identifies the blue whale as the largest.'),
    ('r5', 'Tom studied all week and passed the exam. What helped Tom pass?', 'studying', 'Studying all week is the cause directly linked to passing the exam.'),
    ('r6', 'Plants use sunlight to make food. What energy source do plants use?', 'sunlight', 'The text states "sunlight" directly as the energy source.'),
    ('r7', 'The café was crowded, yet we found one empty table. Was seating available?', 'yes', 'One empty table was found, so seating was available.'),
    ('r8', 'Lena whispered because the baby was sleeping. Why did Lena speak quietly?', 'the baby was sleeping', 'The "because" clause directly gives the reason for whispering.'),
    ('r9', 'After the storm, several roads were flooded. What caused the road closures?', 'flooding', 'Flooding after the storm caused the road closures.'),
    ('r10', 'The recipe serves four people. We have eight guests. How many batches are needed?', 'two', '8 ÷ 4 = 2, so two batches are needed.'),
    ('r11', 'Although the task was difficult, Rui finished it. Did difficulty stop Rui?', 'no', '"Although" signals contrast — Rui completed the task despite the difficulty.'),
    ('r12', 'The museum is free on Sundays. Today is Sunday. What is today\'s admission cost?', 'free', 'The text states the museum is free on Sundays.'),
]


def _vocabulary_questions(prefix='vocab', *, answer_mode='multiple_choice', difficulty='intermediate'):
    return [
        _question(
            f'{prefix}-{word}', prompt, word,
            f'"{word}" means "{meaning}" and fits naturally in this context.',
            f'Now write one original sentence using "{word}".', section='vocabulary',
            answer_mode=answer_mode, difficulty=difficulty,
        )
        for word, meaning, prompt in VOCABULARY_DATA
    ]


def _grammar_questions(*, answer_mode='typing', difficulty='intermediate'):
    return [
        _question(
            key, prompt, answer, explanation,
            'Write one original sentence using the same grammar rule.',
            section='grammar', answer_mode=answer_mode, difficulty=difficulty,
        )
        for key, prompt, answer, explanation in GRAMMAR_DATA
    ]


def _reading_questions(*, answer_mode='typing', difficulty='intermediate'):
    return [
        _question(
            key, prompt, answer, explanation,
            'Identify the exact words in the passage that support your answer.',
            section='reading', answer_mode=answer_mode, difficulty=difficulty,
        )
        for key, prompt, answer, explanation in READING_DATA
    ]


COURSES = {
    'daily-english': {
        'title': 'Daily English Essentials',
        'description': 'Learn essential everyday vocabulary in context.',
        'questions': _vocabulary_questions('daily')[:10],
    },
    'academic-words': {
        'title': 'Academic Word Builder',
        'description': 'Build academic vocabulary for reports and presentations.',
        'questions': _vocabulary_questions('academic')[3:13],
    },
    'business-context': {
        'title': 'Business Vocabulary',
        'description': 'Practice business vocabulary with real-world example sentences.',
        'questions': _vocabulary_questions('business')[5:15],
    },
}


WORLD_ONE_STAGES = {
    f'world-1-stage-{number}': stage
    for number, stage in enumerate((
        {
            'title': 'First Steps', 'level': 'A1', 'difficulty': 'beginner',
            'focus': 'Greetings, numbers, colors, and very common everyday nouns.',
        },
        {
            'title': 'Everyday Town', 'level': 'A1+', 'difficulty': 'beginner',
            'focus': 'Home, school, food, family, and simple present-tense context.',
        },
        {
            'title': 'Context Hills', 'level': 'A2', 'difficulty': 'beginner',
            'focus': 'Short contextual sentences, common verbs, adjectives, and prepositions.',
        },
        {
            'title': 'Collocation Bridge', 'level': 'A2+', 'difficulty': 'intermediate',
            'focus': 'Common collocations, word families, and two-clause everyday sentences.',
        },
        {
            'title': 'Meaning Forest', 'level': 'B1', 'difficulty': 'intermediate',
            'focus': 'Vocabulary in context, phrasal verbs, and choosing between close meanings.',
        },
        {
            'title': 'Grammar Cavern', 'level': 'B1+', 'difficulty': 'intermediate',
            'focus': 'Word form, tense agreement, connectors, and multi-clause context.',
        },
        {
            'title': 'Inference Heights', 'level': 'B2', 'difficulty': 'intermediate',
            'focus': 'Academic vocabulary, inference, tone, and less explicit context clues.',
        },
        {
            'title': 'Nuance Skyway', 'level': 'B2+', 'difficulty': 'advanced',
            'focus': 'Nuanced synonyms, register, complex grammar, and abstract topics.',
        },
        {
            'title': 'Expert Fortress', 'level': 'C1', 'difficulty': 'advanced',
            'focus': 'Dense academic context, idiomatic usage, and subtle distractors.',
        },
        {
            'title': 'World 1 Final', 'level': 'C1+', 'difficulty': 'advanced',
            'focus': 'A cumulative mastery challenge with nuanced vocabulary and deep context.',
        },
    ), start=1)
}

READING_WORLD_STAGES = {
    f'reading-world-1-stage-{number}': stage
    for number, stage in enumerate((
        {
            'title': 'Story Start', 'level': 'A1', 'difficulty': 'beginner',
            'focus': 'Find people, places, times, and other directly stated details in very short passages.',
        },
        {
            'title': 'Sequence Road', 'level': 'A1+', 'difficulty': 'beginner',
            'focus': 'Follow the order of simple events and identify basic who, what, and where information.',
        },
        {
            'title': 'Main-Idea Meadow', 'level': 'A2', 'difficulty': 'beginner',
            'focus': 'Choose the main idea and supporting details in short everyday texts.',
        },
        {
            'title': 'Cause Bridge', 'level': 'A2+', 'difficulty': 'intermediate',
            'focus': 'Recognize cause and effect, reference words, and simple paraphrases.',
        },
        {
            'title': 'Context Forest', 'level': 'B1', 'difficulty': 'intermediate',
            'focus': 'Infer word meaning from context and connect evidence across a paragraph.',
        },
        {
            'title': 'Inference Cave', 'level': 'B1+', 'difficulty': 'intermediate',
            'focus': 'Make supported inferences and distinguish facts from implied information.',
        },
        {
            'title': 'Contrast Cliffs', 'level': 'B2', 'difficulty': 'intermediate',
            'focus': 'Compare viewpoints, track arguments, and identify relationships across longer texts.',
        },
        {
            'title': 'Purpose Skyway', 'level': 'B2+', 'difficulty': 'advanced',
            'focus': 'Analyze tone, register, author purpose, and subtle supporting evidence.',
        },
        {
            'title': 'Evidence Fortress', 'level': 'C1', 'difficulty': 'advanced',
            'focus': 'Evaluate implicit claims, assumptions, evidence quality, and nuanced arguments.',
        },
        {
            'title': 'Reading Final', 'level': 'C1+', 'difficulty': 'advanced',
            'focus': 'Synthesize complex information and evaluate tone, purpose, inference, and evidence together.',
        },
    ), start=1)
}

GRAMMAR_WORLD_STAGES = {
    f'grammar-world-1-stage-{number}': stage
    for number, stage in enumerate((
        {
            'title': 'Sentence Start', 'level': 'A1', 'difficulty': 'beginner',
            'focus': 'Be verbs, simple present tense, basic word order, and subject-verb agreement.',
        },
        {
            'title': 'Building Blocks', 'level': 'A1+', 'difficulty': 'beginner',
            'focus': 'Articles, plurals, pronouns, possessives, and basic questions and negatives.',
        },
        {
            'title': 'Tense Trail', 'level': 'A2', 'difficulty': 'beginner',
            'focus': 'Present and past forms, prepositions, adverbs, and common sentence patterns.',
        },
        {
            'title': 'Modal Bridge', 'level': 'A2+', 'difficulty': 'intermediate',
            'focus': 'Future forms, comparatives, superlatives, modal verbs, and conjunctions.',
        },
        {
            'title': 'Perfect Forest', 'level': 'B1', 'difficulty': 'intermediate',
            'focus': 'Perfect tenses, gerunds, infinitives, and choosing tense from context.',
        },
        {
            'title': 'Clause Cavern', 'level': 'B1+', 'difficulty': 'intermediate',
            'focus': 'Conditionals, passive voice, relative clauses, and multi-clause agreement.',
        },
        {
            'title': 'Reported Heights', 'level': 'B2', 'difficulty': 'intermediate',
            'focus': 'Reported speech, complex connectors, participle clauses, and tense consistency.',
        },
        {
            'title': 'Inversion Skyway', 'level': 'B2+', 'difficulty': 'advanced',
            'focus': 'Inversion, subjunctive forms, advanced conditionals, and reduced clauses.',
        },
        {
            'title': 'Nuance Fortress', 'level': 'C1', 'difficulty': 'advanced',
            'focus': 'Nuanced tense and aspect, ellipsis, emphasis, agreement, and stylistic choices.',
        },
        {
            'title': 'Grammar Final', 'level': 'C1+', 'difficulty': 'advanced',
            'focus': 'A cumulative editing and sentence-transformation challenge using complex grammar.',
        },
    ), start=1)
}

SECTION_STAGE_WORLDS = {
    'vocabulary': WORLD_ONE_STAGES,
    'reading': READING_WORLD_STAGES,
    'grammar': GRAMMAR_WORLD_STAGES,
}
ALL_STAGES = {
    slug: {**stage, 'section': section}
    for section, stages in SECTION_STAGE_WORLDS.items()
    for slug, stage in stages.items()
}

MYSELF_STAGE_DEFINITIONS = (
    {
        'title': 'Material Start', 'level': 'FOUNDATION', 'difficulty': 'beginner',
        'focus': 'Direct facts, essential terms, and the simplest concepts in the uploaded material.',
    },
    {
        'title': 'Key Ideas', 'level': 'BASIC', 'difficulty': 'beginner',
        'focus': 'Important details, definitions, and relationships between the main ideas.',
    },
    {
        'title': 'Context Challenge', 'level': 'INTERMEDIATE', 'difficulty': 'intermediate',
        'focus': 'Use context, compare ideas, and explain information from different parts of the material.',
    },
    {
        'title': 'Applied Practice', 'level': 'APPLIED', 'difficulty': 'intermediate',
        'focus': 'Apply the uploaded material to new examples, situations, and multi-step questions.',
    },
    {
        'title': 'Material Final', 'level': 'CHALLENGE', 'difficulty': 'advanced',
        'focus': 'A cumulative mastery challenge requiring inference, application, and careful distinctions.',
    },
)

STAGE_CLEAR_PERCENT = 70


def stage_catalog(session_key, section='vocabulary'):
    """Return World 1 stages with sequential unlocking and saved best scores."""
    stages = SECTION_STAGE_WORLDS.get(section, {})
    progress = {
        item.course_slug: item
        for item in LanguageCourseProgress.objects.filter(
            browser_session_key=session_key,
            course_slug__in=stages,
        )
    }
    catalog = []
    previous_completed = True
    for index, (slug, stage) in enumerate(stages.items(), start=1):
        item = progress.get(slug)
        completed = bool(item and item.completed)
        catalog.append({
            'slug': slug,
            'number': f'1-{index}',
            'rank': index,
            'title': stage['title'],
            'level': stage['level'],
            'difficulty': stage['difficulty'],
            'focus': stage['focus'],
            'score_percent': item.score_percent if item else 0,
            'completed': completed,
            'unlocked': previous_completed or completed,
            'boss': index == len(stages),
        })
        previous_completed = completed
    return catalog


def stage_is_unlocked(session_key, stage_slug):
    stage = ALL_STAGES.get(stage_slug)
    if stage is None:
        return False
    return any(
        item['slug'] == stage_slug and item['unlocked']
        for item in stage_catalog(session_key, stage['section'])
    )


def myself_stage_slug(pack_id, stage_number):
    pack_hex = getattr(pack_id, 'hex', str(pack_id).replace('-', ''))
    return f'myself-{pack_hex}-stage-{stage_number}'


def myself_pack_id_from_stage_slug(course_slug):
    match = re.fullmatch(r'myself-([0-9a-f]{32})-stage-[1-5]', course_slug or '')
    if not match:
        return ''
    value = match.group(1)
    return f'{value[:8]}-{value[8:12]}-{value[12:16]}-{value[16:20]}-{value[20:]}'


def myself_stage_catalog(pack, session_key):
    slugs = [myself_stage_slug(pack.id, index) for index in range(1, 6)]
    progress = {
        item.course_slug: item
        for item in LanguageCourseProgress.objects.filter(
            browser_session_key=session_key,
            course_slug__in=slugs,
        )
    }
    catalog = []
    previous_completed = True
    for index, definition in enumerate(MYSELF_STAGE_DEFINITIONS, start=1):
        slug = slugs[index - 1]
        item = progress.get(slug)
        completed = bool(item and item.completed)
        catalog.append({
            'slug': slug,
            'number': f'1-{index}',
            'rank': index,
            **definition,
            'score_percent': item.score_percent if item else 0,
            'completed': completed,
            'unlocked': previous_completed or completed,
            'boss': index == len(MYSELF_STAGE_DEFINITIONS),
        })
        previous_completed = completed
    return catalog


def course_catalog(session_key):
    progress = {
        item.course_slug: item
        for item in LanguageCourseProgress.objects.filter(browser_session_key=session_key)
    }
    catalog = []
    recommended_difficulty = difficulty_from_diagnostic(session_key, 'vocabulary')
    for slug, course in COURSES.items():
        item = progress.get(slug)
        catalog.append({
            'slug': slug,
            'title': course['title'],
            'description': course['description'],
            'score_percent': item.score_percent if item else 0,
            'completed': item.completed if item else False,
            'difficulty': recommended_difficulty,
        })
    return catalog


def difficulty_from_diagnostic(session_key, section):
    """Map the latest completed diagnostic's section score to a quiz level."""
    if section not in {'vocabulary', 'grammar', 'reading'}:
        return 'intermediate'
    run = LanguageQuizRun.objects.filter(
        browser_session_key=session_key,
        section='diagnostic',
        finished=True,
    ).first()
    if run is None:
        return 'intermediate'
    relevant = [q for q in run.questions if q.get('section') == section]
    if not relevant:
        return 'intermediate'
    score = sum(q.get('is_correct') is True for q in relevant) / len(relevant)
    if score < 0.4:
        return 'beginner'
    if score < 0.8:
        return 'intermediate'
    return 'advanced'


def _questions_from_ai(specs, section, count, *, answer_mode='multiple_choice', difficulty='intermediate'):
    # Providers answer in the AI_QUESTION_SCHEMA shape ({"questions": [...]});
    # a bare list is still accepted for callers/tests that pass one directly.
    if isinstance(specs, dict):
        specs = specs.get('questions')
    if not isinstance(specs, list) or len(specs) < count:
        return []
    questions = []
    for index, spec in enumerate(specs[:count]):
        if not isinstance(spec, dict) or not all(spec.get(key) for key in ('prompt', 'answer', 'explanation')):
            return []
        hints = spec.get('hints') if isinstance(spec.get('hints'), list) else []
        if len(hints) < 5:
            hints = None
        key = hashlib.sha256(
            f"ai|{section}|{spec['prompt']}|{spec['answer']}|{index}".encode('utf-8')
        ).hexdigest()[:24]
        item_section = ('vocabulary', 'grammar', 'reading')[index % 3] if section == 'diagnostic' else section
        q = _question(
            key,
            str(spec['prompt']),
            str(spec['answer']),
            str(spec['explanation']),
            str(spec.get('next_step') or 'Write one original sentence using the same concept.'),
            section=item_section,
            hints=hints,
            answer_mode=(
                answer_mode
                if section != 'diagnostic' and item_section in {'vocabulary', 'reading', 'grammar', 'myself'}
                else answer_mode if item_section == 'vocabulary'
                else 'typing'
            ),
            choices=spec.get('choices') if isinstance(spec.get('choices'), list) else None,
            difficulty=difficulty,
        )
        # Carry the skill_focus label into the question dict for display in the UI
        q['skill_focus'] = str(spec.get('skill_focus') or '')
        questions.append(q)
    return questions


def generate_section_questions(
    section, *, count=10, answer_mode='multiple_choice',
    difficulty='intermediate', course_context='', fallback_questions=None,
):
    if answer_mode not in VOCABULARY_ANSWER_MODES:
        raise ValueError('Unsupported vocabulary answer mode.')
    if difficulty not in DIFFICULTY_LABELS:
        difficulty = 'intermediate'
    if section in {'vocabulary', 'grammar', 'reading', 'diagnostic'}:
        try:
            specs = generate_ai_response(
                system_prompt=(
                    'You are an expert English language teacher creating quiz questions. '
                    'Return exactly the requested number of well-crafted questions. Each question must have: '
                    'prompt (the question text), '
                    'answer (short, unambiguous correct answer), '
                    'skill_focus (the specific language skill tested, e.g. "Past Tense", "Inference", "Vocabulary in Context"), '
                    'explanation (clear English explanation of why the answer is correct and what rule/principle applies), '
                    'next_step (one actionable study task in English for the learner to practise further), '
                    'hints (exactly 5 English hints, each narrowing the possible answer more than the previous one, '
                    'with the full answer only at level 5), and choices. '
                    'For vocabulary, reading, or grammar multiple_choice questions, choices must contain exactly 5 distinct '
                    'options including the answer. Distractors must be plausible but clearly less suitable than the answer. '
                    'For typing questions, choices must be an empty array.'
                ),
                user_prompt=(
                    f'Section: {section}. Difficulty: {difficulty}. Create {count} fresh questions. '
                    f'Difficulty requirements: {DIFFICULTY_GUIDANCE[difficulty]} '
                    f'Course theme: {course_context or "General English practice"}. '
                    f'Answer mode: {answer_mode}. '
                    'For diagnostic, mix vocabulary, grammar, and short reading questions. '
                    'Avoid duplicates and keep prompts self-contained.'
                ),
                response_schema=AI_QUESTION_SCHEMA,
            )
            generated = _questions_from_ai(
                specs, section, count, answer_mode=answer_mode, difficulty=difficulty,
            )
            if generated:
                return generated
        except AIEngineError:
            pass

    if fallback_questions is not None:
        pool = []
        for original in fallback_questions:
            question = dict(original)
            question['difficulty'] = difficulty
            question['hints'] = list(original.get('hints', []))
            question['choices'] = list(original.get('choices', []))
            pool.append(question)
    elif section == 'vocabulary':
        pool = _vocabulary_questions(answer_mode=answer_mode, difficulty=difficulty)
    elif section == 'grammar':
        pool = _grammar_questions(answer_mode=answer_mode, difficulty=difficulty)
    elif section == 'reading':
        pool = _reading_questions(answer_mode=answer_mode, difficulty=difficulty)
    elif section == 'diagnostic':
        pool = (
            _vocabulary_questions(difficulty=difficulty)
            + _grammar_questions(difficulty=difficulty)
            + _reading_questions(difficulty=difficulty)
        )
    else:
        raise ValueError('Unsupported language section.')
    return random.sample(pool, min(count, len(pool)))


def get_course_questions(course_slug, *, difficulty='intermediate'):
    course = COURSES.get(course_slug)
    if course is None:
        raise ValueError('The selected course was not found.')
    generated = generate_section_questions(
        'vocabulary',
        count=10,
        answer_mode='multiple_choice',
        difficulty=difficulty,
        course_context=f'{course["title"]}: {course["description"]}',
        fallback_questions=course['questions'],
    )
    return generated


def get_stage_questions(stage_slug):
    stage = ALL_STAGES.get(stage_slug)
    if stage is None:
        raise ValueError('The selected stage was not found.')
    section = stage['section']
    stages = SECTION_STAGE_WORLDS[section]
    stage_number = list(stages).index(stage_slug) + 1
    context = (
        f'{section.title()} World 1 Stage 1-{stage_number}: {stage["title"]}. '
        f'Target level: CEFR {stage["level"]}. Focus: {stage["focus"]} '
        f'This is difficulty step {stage_number} of 10, so make it clearly harder than '
        f'step {max(1, stage_number - 1)} while staying within the target level.'
    )
    questions = generate_section_questions(
        section,
        count=10,
        answer_mode='multiple_choice',
        difficulty=stage['difficulty'],
        course_context=context,
    )
    for question in questions:
        original_key = question['key']
        question['key'] = hashlib.sha256(
            f'{stage_slug}|{original_key}'.encode('utf-8')
        ).hexdigest()[:24]
        question['stage_number'] = f'1-{stage_number}'
        question['stage_level'] = stage['level']
        question['stage_section'] = section
    return questions


def _extract_pdf_text(data: bytes) -> str:
    """Extract PDF text while distinguishing setup and file errors."""
    try:
        import io as _io
        from pypdf import PdfReader
    except ImportError as exc:
        raise ValueError(
            'PDF support is not installed. Run "pip install -r requirements.txt" '
            'in the active virtual environment and restart the server.'
        ) from exc

    try:
        reader = PdfReader(_io.BytesIO(data))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ''
            if text.strip():
                pages.append(text)
        return '\n'.join(pages)
    except Exception as exc:
        raise ValueError(
            'The PDF could not be read. It may be damaged, encrypted, or unsupported.'
        ) from exc


def _decode_upload(upload) -> str:
    """Read an uploaded file and return its plain text content.

    Handles PDF via pypdf, JSON/CSV/text via encoding detection.
    Falls back to latin-1 byte decoding as a last resort for unknown formats.
    """
    data = upload.read()
    name = upload.name.lower()

    # PDF: use a proper PDF parser — never decode binary PDF as text
    if name.endswith('.pdf'):
        text = _extract_pdf_text(data)
        if text.strip():
            return text
        # PDF has no extractable text (e.g. scanned image-only PDF)
        raise ValueError(
            'The PDF appears to be a scanned image with no extractable text. '
            'Please upload a text-based PDF, TXT, or Markdown file.'
        )

    # Plain text files: try common encodings
    for encoding in ('utf-8-sig', 'utf-8', 'utf-16', 'shift-jis'):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return data.decode('latin-1', errors='ignore')


_EXACT_ANSWER_KEY_HEADING = re.compile(
    r'(?im)^(?:解答一覧|answer\s+key|answers?)\s*$'
)
_EXACT_NUMBERED_LINE = re.compile(r'(?m)^(?P<number>\d{1,3})\.\s*(?P<text>.+?)\s*$')
_EXACT_CHOICE_LINE = re.compile(r'^([A-E])\.\s*(.+?)\s*$')
_EXACT_ANSWER_LINE = re.compile(r'(?m)^(\d{1,3})\.\s*([A-E])(?:\.|\s)+(.+?)\s*$')


def _parse_exact_pdf_quiz(text, *, section='vocabulary', source_name='uploaded.pdf'):
    """Parse a numbered five-choice PDF quiz without rewriting its content."""
    heading = _EXACT_ANSWER_KEY_HEADING.search(text or '')
    if not heading:
        raise ValueError(
            '解答一覧を検出できませんでした。「解答一覧」または「Answer Key」を含むPDFを選択してください。'
        )

    question_text = text[:heading.start()]
    answer_text = text[heading.end():]
    answers = {}
    for match in _EXACT_ANSWER_LINE.finditer(answer_text):
        number = int(match.group(1))
        if number in answers:
            raise ValueError(f'解答一覧に問題{number}の解答が重複しています。')
        answers[number] = (match.group(2), match.group(3).strip())

    matches = list(_EXACT_NUMBERED_LINE.finditer(question_text))
    if not matches:
        raise ValueError('PDFから番号付きの問題を検出できませんでした。')

    parsed = []
    seen_numbers = set()
    for index, match in enumerate(matches):
        number = int(match.group('number'))
        if number in seen_numbers:
            raise ValueError(f'問題番号{number}が重複しています。')
        seen_numbers.add(number)
        block_end = matches[index + 1].start() if index + 1 < len(matches) else len(question_text)
        lines = [line.strip() for line in question_text[match.end():block_end].splitlines() if line.strip()]
        choices_by_letter = {}
        context_lines = []
        for line in lines:
            choice_match = _EXACT_CHOICE_LINE.match(line)
            if choice_match:
                letter, choice_text = choice_match.groups()
                choices_by_letter[letter] = f'{letter}. {choice_text}'
            elif not choices_by_letter:
                context_lines.append(line)

        if list(choices_by_letter) != list('ABCDE'):
            raise ValueError(f'問題{number}の選択肢A〜Eを正しく検出できませんでした。')
        if number not in answers:
            raise ValueError(f'解答一覧に問題{number}の解答がありません。')

        answer_letter, answer_text_value = answers[number]
        answer_choice = choices_by_letter.get(answer_letter)
        if not answer_choice:
            raise ValueError(f'問題{number}の正解記号{answer_letter}が選択肢にありません。')
        choice_value = answer_choice.split('.', 1)[1].strip()
        if choice_value.casefold() != answer_text_value.casefold():
            raise ValueError(
                f'問題{number}の選択肢と解答一覧が一致しません。'
            )

        original_prompt = f'{match.group("number")}. {match.group("text")}'
        if context_lines:
            original_prompt += '\n' + '\n'.join(context_lines)
        choices = [choices_by_letter[letter] for letter in 'ABCDE']
        key = hashlib.sha256(
            f'exact-pdf|{source_name}|{number}|{original_prompt}|{answer_choice}'.encode('utf-8')
        ).hexdigest()[:24]
        question = _question(
            key,
            original_prompt,
            answer_choice,
            f'PDFの解答一覧では「{answer_choice}」が正解です。',
            '正解語を入れた文を声に出して読み、文脈と語彙を確認しましょう。',
            section=section,
            answer_mode='multiple_choice',
            choices=choices,
        )
        question['skill_focus'] = 'PDF原文問題'
        question['source_number'] = number
        question['source_answer_letter'] = answer_letter
        parsed.append(question)

    if set(answers) != seen_numbers:
        missing_questions = sorted(set(answers) - seen_numbers)
        raise ValueError(
            f'解答一覧に対応する問題がありません: {missing_questions}'
        )
    return parsed


def import_pdf_questions_exact(files, *, section='vocabulary'):
    """Import one text-based PDF quiz exactly, without calling AI."""
    files = list(files)
    if len(files) != 1 or not files[0].name.lower().endswith('.pdf'):
        raise ValueError('そのまま出題モードではPDFを1ファイルだけ選択してください。')
    upload = files[0]
    text = _decode_upload(upload)
    questions = _parse_exact_pdf_quiz(
        text, section=section, source_name=upload.name,
    )
    return questions, upload.name


_UPLOAD_SYSTEM_PROMPTS = {
    'reading': (
        'You are an expert English reading teacher designing questions for language learners. '
        'Create questions STRICTLY based on the supplied passage. '
        'Distribute the questions across these reading skill levels (Bloom\'s Taxonomy): '
        '- 2 LITERAL questions: ask who/what/where/when using exact words from the text. '
        '- 3 INFERENCE questions: ask why/how/what-does-this-imply — answer must be derived from clues in the text, not stated directly. '
        '- 2 MAIN IDEA questions: ask about the central message, topic, or purpose of a paragraph. '
        '- 2 VOCABULARY-IN-CONTEXT questions: ask what a specific word or phrase means AS USED in the passage (not a dictionary definition). '
        '- 1 CAUSE-AND-EFFECT question: ask what caused X, or what was the result of Y. '
        'Rules: '
        '(1) Every answer must be a short phrase (1–6 words). '
        '(2) Answers must be clearly supported by the text. '
        '(3) skill_focus must be one of: Literal Detail, Inference, Main Idea, Vocabulary in Context, Cause and Effect. '
        '(4) explanation must quote the relevant sentence from the passage and explain why that answer is correct. '
        '(5) next_step must ask the learner to find and quote the supporting sentence from the text. '
        '(6) hints must be exactly 5, progressing: '
        '  hint1=point to the relevant paragraph, '
        '  hint2=point to the relevant sentence, '
        '  hint3=give a conceptual clue about the answer type, '
        '  hint4=give the first word of the answer, '
        '  hint5=state the full answer. '
        'Return exactly the requested count in the "questions" array.'
    ),
    'grammar': (
        'You are an expert English grammar teacher designing questions for language learners. '
        'Create questions STRICTLY based on sentences from the supplied text. '
        'Mix these question types evenly: '
        '- FILL-IN-THE-BLANK: take a sentence from the text, remove one grammatical element, ask the learner to fill it in. '
        '- ERROR CORRECTION: take a sentence from the text, introduce one grammatical error, ask the learner to identify and correct it. '
        '- SENTENCE TRANSFORMATION: give a sentence from the text and ask the learner to rewrite it in another form (e.g., active→passive, affirmative→negative, present→past). '
        'Target grammar points present in the material such as: verb tenses, subject-verb agreement, articles (a/an/the), '
        'prepositions, passive voice, relative clauses, conditionals, modal verbs, gerunds vs infinitives, '
        'comparatives/superlatives, reported speech, conjunctions. '
        'Rules: '
        '(1) skill_focus must name the exact grammar rule being tested, e.g. "Present Perfect Tense", "Passive Voice", "Third Conditional". '
        '(2) prompt must clearly show the question type (e.g., start with "Fill in the blank:", "Correct the error:", "Transform the sentence:"). '
        '(3) answer must be the corrected or completed text (short phrase or full clause). '
        '(4) explanation must state the grammar RULE explicitly (e.g., "The present perfect uses have/has + past participle to...") and then show why the answer applies. '
        '(5) next_step must ask the learner to write one original sentence using the same grammar rule. '
        '(6) hints must be exactly 5, progressing: '
        '  hint1=name the grammar category (e.g., "Think about verb tense here."), '
        '  hint2=state the grammar rule in simple terms, '
        '  hint3=give the sentence structure/formula (e.g., "Subject + have/has + past participle"), '
        '  hint4=give the first word(s) of the answer, '
        '  hint5=state the full answer. '
        'Return exactly the requested count in the "questions" array.'
    ),
    'vocabulary': (
        'You are an expert English vocabulary teacher. '
        'Create vocabulary questions ONLY from the supplied learning material. '
        'Follow the learner instruction. '
        'For each question: skill_focus must name the vocabulary strategy used '
        '(e.g., "Context Clues", "Word Form", "Collocations", "Synonyms/Antonyms"). '
        'Return exactly the requested count. Each item must have one concise answer, '
        'a clear English explanation showing why that word fits, a next study step, '
        'and exactly five English hints that progress from subtle to answer-revealing at level 5.'
    ),
    'myself': (
        'You create language-learning questions only from the supplied learning material. '
        'Treat both the supplied material and the learner instruction as mandatory context. '
        'Follow the learner instruction. Return exactly the requested count. Each item must '
        'have one concise answer, a clear English explanation, next study step, skill_focus label, '
        'and exactly five English hints that progress from subtle to revealing the answer at level 5. '
        'When answer mode is multiple_choice, analyze the material and learner instruction to create '
        'exactly five choices: one uniquely correct answer and four plausible distractors. Every distractor '
        'must match the question type and semantic category, and must represent a believable misunderstanding '
        'of the uploaded content. Do not use unrelated words, arbitrary answers, "all of the above", '
        '"none of the above", or "not stated" unless the learner explicitly requests them.'
    ),
}


def extract_material(files):
    """Extract and normalize uploaded material once for AI generation and reuse."""
    chunks = []
    names = []
    for upload in files:
        names.append(upload.name)
        text = _decode_upload(upload)
        if upload.name.lower().endswith('.json'):
            try:
                text = json.dumps(json.loads(text), ensure_ascii=False)
            except json.JSONDecodeError:
                pass
        elif upload.name.lower().endswith('.csv'):
            rows = csv.reader(io.StringIO(text))
            text = '\n'.join(' : '.join(cell.strip() for cell in row if cell.strip()) for row in rows)
        chunks.extend(line.strip() for line in text.splitlines() if line.strip())

    if not chunks:
        raise ValueError('No readable text could be extracted from the uploaded file(s).')

    return '\n'.join(chunks)[:16000], ', '.join(names), chunks


def generate_uploaded_questions(
    files, instruction, *, section='myself', count=10,
    answer_mode='multiple_choice', difficulty='intermediate',
):
    """Create questions from user-uploaded material using section-aware AI prompts."""
    material, source_name, chunks = extract_material(files)

    system_prompt = _UPLOAD_SYSTEM_PROMPTS.get(section, _UPLOAD_SYSTEM_PROMPTS['myself'])
    try:
        specs = generate_ai_response(
            system_prompt=system_prompt,
            user_prompt=(
                f'Learner instruction: {instruction}\n'
                f'Question count: {count}\n'
                f'Difficulty: {difficulty}\n'
                f'Answer mode: {answer_mode}. '
                'For multiple_choice use exactly 5 distinct choices including the answer; '
                'otherwise return an empty choices array.\n'
                f'Material:\n{material}'
            ),
            response_schema=AI_QUESTION_SCHEMA,
        )
        generated = _questions_from_ai(
            specs, section, count, answer_mode=answer_mode, difficulty=difficulty,
        )
        if generated and section == 'myself' and answer_mode == 'multiple_choice':
            valid_choices = all(
                len(question.get('choices', [])) == 5
                and len({str(choice).strip().casefold() for choice in question['choices']}) == 5
                and str(question['answer']).strip().casefold() in {
                    str(choice).strip().casefold() for choice in question['choices']
                }
                for question in generated
            )
            if not valid_choices:
                raise ValueError(
                    'AIが教材と指示に沿った有効な5択を作成できませんでした。'
                    '指示を具体的にして、もう一度作成してください。'
                )
        if generated:
            return generated, source_name
        if section == 'myself' and answer_mode == 'multiple_choice':
            raise ValueError(
                'AIが10問の有効な5択問題を返しませんでした。もう一度作成してください。'
            )
    except AIEngineError as exc:
        if section == 'myself' and answer_mode == 'multiple_choice':
            raise ValueError(
                'AIへ接続できないため、教材に沿った5択を作成できませんでした。'
                'AIプロバイダー（DeepSeek / Gemini / Groq）の設定と接続を確認して、もう一度お試しください。'
            ) from exc

    # --- Smart rule-based fallback (when AI is unavailable) ---
    # Pick the longest, most content-rich sentences as source material
    sentences = [
        s.strip() for s in re.split(r'(?<=[.!?])\s+', '\n'.join(chunks))
        if len(s.strip()) > 40
    ]
    if not sentences:
        sentences = [c for c in chunks if len(c) > 20] or chunks

    random.shuffle(sentences)
    questions = []

    for index in range(count):
        sentence = sentences[index % len(sentences)]
        words = re.findall(r'\b[A-Za-z]{4,}\b', sentence)
        key = hashlib.sha256(f'{sentence}|{index}|{section}'.encode('utf-8')).hexdigest()[:20]

        if section == 'reading':
            # Reading fallback: ask a genuine comprehension question about the sentence
            skill_types = [
                'main_idea', 'detail', 'inference', 'vocab', 'cause_effect'
            ]
            skill = skill_types[index % len(skill_types)]

            if skill == 'main_idea':
                q_prompt = f'What is the main idea expressed in this sentence from the passage?\n\n"{sentence}"'
                q_answer = words[-1].lower() if words else sentence.split()[-1].lower()
                skill_label = 'Main Idea'
                explanation = f'The sentence "{ sentence }" expresses its central point through the word "{q_answer}".'
                next_step_text = 'Re-read the surrounding paragraph and identify which sentence best supports this idea.'
                hints = [
                    'Read the full sentence carefully and identify its subject.',
                    'Ask yourself: what is this sentence primarily about?',
                    f'Focus on the final part of the sentence: "...{sentence[-50:]}"',
                    f'The answer is a word found near the end of the sentence.',
                    f'The answer is "{q_answer}".'
                ]
            elif skill == 'detail':
                target = words[0] if words else sentence.split()[0]
                q_prompt = f'According to the passage, complete the detail: "{sentence.replace(target, "____", 1)}"'
                q_answer = target.lower()
                skill_label = 'Literal Detail'
                explanation = f'The text states this directly: "{sentence}". The missing word is "{target}".'
                next_step_text = 'Find and quote the exact sentence from the passage that confirms this detail.'
                hints = [
                    'Look at the beginning of the sentence for context.',
                    'The answer is a specific word used directly in the passage.',
                    f'The sentence starts with: "{sentence[:40]}..."',
                    f'The answer has {len(target)} letters and starts with "{target[0]}".'
                    f' The answer is "{q_answer}".'
                ]
            elif skill == 'vocab':
                target = max(words, key=len) if words else 'word'
                q_prompt = f'What does the word "{target}" mean as used in this sentence?\n\n"{sentence}"'
                q_answer = target.lower()
                skill_label = 'Vocabulary in Context'
                explanation = f'In context, "{target}" refers to the concept described in the sentence: "{sentence}".'
                next_step_text = f'Use "{target}" in a new sentence with a different subject.'
                hints = [
                    f'Read the whole sentence and think about what "{target}" is doing grammatically.',
                    'Is it a noun, verb, or adjective? That helps narrow the meaning.',
                    f'Look at the words before and after "{target}" for clues.',
                    f'The answer is the word itself: it starts with "{target[0]}".'
                    f' The answer is "{q_answer}".'
                ]
            elif skill == 'cause_effect':
                q_prompt = f'What is the likely cause or result described in this sentence?\n\n"{sentence}"'
                q_answer = (words[-1] if words else 'result').lower()
                skill_label = 'Cause and Effect'
                explanation = f'The sentence "{sentence}" describes a cause-and-effect relationship. The key outcome is "{q_answer}".'
                next_step_text = 'Find another sentence in the passage that shows a cause-and-effect relationship.'
                hints = [
                    'Look for signal words like "because", "therefore", "as a result", "so".'
                    ' These words mark cause-effect relationships.',
                    'Identify what happened and why it happened in the sentence.',
                    f'Focus on the second half of the sentence: "...{sentence[-60:]}"',
                    f'The key term starts with "{q_answer[0]}".'
                    f' The answer is "{q_answer}".'
                ]
            else:  # inference
                q_prompt = f'Based on this sentence, what can you infer about the topic?\n\n"{sentence}"'
                q_answer = (words[len(words) // 2] if words else 'implied').lower()
                skill_label = 'Inference'
                explanation = f'Although not stated directly, the sentence "{sentence}" implies that "{q_answer}" is a key concept.'
                next_step_text = 'Explain in one sentence why you think this inference is supported by the text.'
                hints = [
                    'Inferences are not stated directly — you need to "read between the lines".',
                    'Think about what the author assumes the reader already knows.',
                    f'Look at the whole sentence and consider its broader context.',
                    f'The implied concept starts with "{q_answer[0]}".'
                    f' The answer is "{q_answer}".'
                ]

        elif section == 'vocabulary':
            target = max(words, key=len) if words else sentence.split()[0]
            q_prompt = f'Choose or type the word that completes this material excerpt:\n\n"{sentence.replace(target, "____", 1)}"'
            q_answer = target.lower()
            skill_label = 'Vocabulary in Context'
            explanation = f'The original material uses "{target}" in this context: "{sentence}"'
            next_step_text = f'Write a new sentence using "{target}".'
            hints = [
                'Use the surrounding words to decide what meaning is needed.',
                'Identify the missing word\'s part of speech.',
                f'The answer has {len(target)} letters.',
                f'The answer begins with "{target[0]}" and ends with "{target[-1]}".',
                f'The answer is "{q_answer}".',
            ]
        else:  # grammar / general section
            # Grammar fallback: create fill-in-the-blank on actual grammatical words
            # Target prepositions, auxiliary verbs, articles, conjunctions — real grammar words
            grammar_targets = re.findall(
                r'\b(is|are|was|were|has|have|had|will|would|could|should|may|might'
                r'|be|been|being|do|does|did'
                r'|a|an|the'
                r'|in|on|at|by|for|with|about|from|to|of|into|through|during|before|after'
                r'|because|although|however|therefore|since|while|when|if|unless|until'
                r'|which|who|that|whose)\b',
                sentence, flags=re.IGNORECASE
            )
            if grammar_targets:
                target = grammar_targets[0]
                blanked = re.sub(r'\b' + re.escape(target) + r'\b', '____', sentence, count=1, flags=re.IGNORECASE)
                q_prompt = f'Fill in the blank with the correct word:\n\n"{blanked}"'
                q_answer = target.lower()
                skill_label = 'Grammar: Function Word'
                explanation = (
                    f'The correct word is "{target}". In the sentence "{sentence}", '
                    f'it functions as a grammatical element that connects or qualifies other parts of the sentence.'
                )
                next_step_text = f'Write a new sentence using "{target}" in the same grammatical role.'
                hints = [
                    'Focus on the grammatical structure of the sentence — what type of word is missing? (preposition, article, conjunction, auxiliary verb)',
                    'Look at what comes before and after the blank to determine the word type needed.',
                    f'The sentence structure suggests a {"preposition" if target.lower() in "in on at by for with about from to of into through during before after" else "function word"} is needed.',
                    f'The answer starts with "{target[0]}" and has {len(target)} letters.',
                    f'The answer is "{q_answer}".'
                ]
            else:
                # Last-resort: find a content word to blank
                target = max(words, key=len) if words else 'concept'
                blanked = sentence.replace(target, '____', 1)
                q_prompt = f'Fill in the blank with the correct word from the passage:\n\n"{blanked}"'
                q_answer = target.lower()
                skill_label = 'Grammar: Vocabulary'
                explanation = f'The word "{target}" completes the sentence correctly: "{sentence}"'
                next_step_text = f'Use "{target}" in a new sentence that demonstrates the same meaning.'
                hints = [
                    'Read the full sentence and think about what part of speech fits in the blank.',
                    f'The sentence context is: "{sentence[:60]}..."',
                    f'The answer has {len(target)} letters.',
                    f'The answer starts with "{target[0]}".'
                    f' The answer is "{q_answer}".'
                ]

        questions.append(_question(
            f'upload-{key}',
            q_prompt,
            q_answer,
            explanation,
            next_step_text,
            section=section,
            hints=hints[:5],
            answer_mode=(
                answer_mode
                if section in {'vocabulary', 'reading', 'grammar', 'myself'}
                else 'typing'
            ),
            difficulty=difficulty,
        ))
        questions[-1]['skill_focus'] = skill_label

    return questions, source_name


def get_myself_stage_questions(pack, stage_number):
    """Generate and namespace ten stable questions for one uploaded-material stage."""
    if stage_number not in range(1, 6):
        raise ValueError('The selected material stage was not found.')
    definition = MYSELF_STAGE_DEFINITIONS[stage_number - 1]
    stage_instruction = (
        f'{pack.instruction}\n'
        f'Create World 1 Stage 1-{stage_number} of 5. '
        f'Stage focus: {definition["focus"]} '
        f'Make this difficulty step {stage_number} of 5 and avoid questions that belong to an easier step.'
    )
    upload = ContentFile(pack.material_text.encode('utf-8'), name='material.txt')
    questions, _source_name = generate_uploaded_questions(
        [upload],
        stage_instruction,
        section='myself',
        count=10,
        answer_mode=pack.answer_mode,
        difficulty=definition['difficulty'],
    )
    if pack.answer_mode == 'multiple_choice':
        for question in questions:
            random.shuffle(question['choices'])
            question['answer_mode'] = 'multiple_choice'
    else:
        for question in questions:
            question['choices'] = []
            question['answer_mode'] = 'typing'
    for question in questions:
        question['key'] = hashlib.sha256(
            f'{pack.id}|{stage_number}|{question["key"]}'.encode('utf-8')
        ).hexdigest()[:24]
        question['stage_number'] = f'1-{stage_number}'
        question['stage_level_label'] = definition['level']
        question['myself_pack_id'] = str(pack.id)
    return questions


def missing_questions(session_key, *, count=10):
    records = list(MissingLanguageQuestion.objects.filter(browser_session_key=session_key))
    random.shuffle(records)
    return [
        _question(
            record.fingerprint, record.prompt, record.reference_answer,
            record.explanation, record.next_step, section='missing',
            hints=record.hints if record.hints else None,
            choices=record.choices,
            answer_mode='multiple_choice' if record.choices else 'typing',
        )
        for record in records[:count]
    ]


def answer_matches(answer, reference):
    normalize = lambda value: re.sub(r'[\s\W_]+', ' ', str(value).casefold()).strip()
    supplied = normalize(answer)
    expected = normalize(reference)
    return bool(supplied and (supplied == expected or supplied in {part.strip() for part in expected.split('/')}))


def fingerprint(question):
    return hashlib.sha256(f"{question['prompt']}|{question['answer']}".encode('utf-8')).hexdigest()


@transaction.atomic
def remember_missing(session_key, question):
    key = fingerprint(question)
    MissingLanguageQuestion.objects.update_or_create(
        browser_session_key=session_key,
        fingerprint=key,
        defaults={
            'section': question.get('section', ''),
            'prompt': question['prompt'],
            'reference_answer': question['answer'],
            'explanation': question.get('explanation', ''),
            'next_step': question.get('next_step', ''),
            'hints': question.get('hints', []),
            'choices': question.get('choices') or [],
        },
    )


def resolve_missing(session_key, question):
    keys = {question.get('key'), fingerprint(question)}
    MissingLanguageQuestion.objects.filter(
        browser_session_key=session_key,
        fingerprint__in=[key for key in keys if key],
    ).delete()


@transaction.atomic
def update_course_progress(session_key, course_slug, question, *, quiz_run=None):
    if not course_slug:
        return None
    progress, _ = LanguageCourseProgress.objects.get_or_create(
        browser_session_key=session_key,
        course_slug=course_slug,
    )
    keys = set(progress.correct_question_keys)
    keys.add(question['key'])
    progress.correct_question_keys = sorted(keys)
    if course_slug in ALL_STAGES or myself_pack_id_from_stage_slug(course_slug):
        total = len(quiz_run.questions) if quiz_run else 10
        correct_count = quiz_run.correct_count if quiz_run else len(keys)
        run_score = min(100, round(correct_count / total * 100))
        progress.score_percent = max(progress.score_percent, run_score)
        progress.completed = progress.completed or progress.score_percent >= STAGE_CLEAR_PERCENT
    else:
        total = len(COURSES[course_slug]['questions'])
        progress.score_percent = min(100, round(len(keys) / total * 100))
        progress.completed = progress.score_percent == 100
    progress.save()
    return progress


def diagnostic_recommendations(questions):
    wrong_sections = {}
    for question in questions:
        if question.get('is_correct') is False:
            section = question.get('section', 'vocabulary')
            wrong_sections[section] = wrong_sections.get(section, 0) + 1
    if not wrong_sections:
        return [
            {'title': 'Academic Word Builder', 'detail': 'Your foundations look solid — try advancing to higher-level vocabulary.'},
            {'title': 'Long-form Reading', 'detail': 'Practice finding supporting evidence quickly in longer passages.'},
        ]
    weakest = max(wrong_sections, key=wrong_sections.get)
    mapping = {
        'vocabulary': ('Vocabulary', 'Review words alongside example sentences in Missing and the Vocabulary courses.'),
        'grammar': ('Grammar', 'Turn each incorrect grammar rule into an example sentence and try again.'),
        'reading': ('Reading', 'Practice marking conjunctions and evidence sentences while reading.'),
    }
    title, detail = mapping.get(weakest, mapping['vocabulary'])
    return [
        {'title': f'Recommended: {title}', 'detail': detail},
        {'title': 'Related: Missing', 'detail': 'Questions you got wrong this session are saved in Missing. They disappear once you answer correctly.'},
    ]

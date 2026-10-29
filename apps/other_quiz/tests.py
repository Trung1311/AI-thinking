from unittest.mock import patch

from django.test import Client, SimpleTestCase, TestCase
from django.urls import reverse

from .models import Lesson, Question, QuestionAttempt, Subject
from .services import can_evaluate_open_response


class OtherSubjectRouteTests(TestCase):
    def test_home_loads(self):
        self.assertEqual(self.client.get(reverse('other_quiz:home')).status_code, 200)


class QuestionAttemptTests(TestCase):
    """Per-browser answer history, which Question.is_correct alone cannot give
    (it is one shared flag on the question — see models.QuestionAttempt)."""

    def setUp(self):
        subject = Subject.objects.create(id='sub_test', title='History')
        lesson = Lesson.objects.create(
            id='les_test', subject=subject, chapter='1', title='Causes of WWII',
        )
        self.question = Question.objects.create(
            lesson=lesson,
            title='Trigger',
            prompt='Which event began the war in Europe?',
            q_type='MULTIPLE_CHOICE',
            options=['Invasion of Poland', 'Fall of France'],
            correct_answer='Invasion of Poland',
        )
        self.url = reverse(
            'other_quiz:lesson_detail',
            kwargs={'subject_id': subject.id, 'lesson_id': lesson.id},
        )

    def _answer(self, client, choice):
        return client.post(self.url, {
            'question_id': self.question.id,
            'action': 'submit_mc',
            f'question_{self.question.id}': choice,
        })

    def test_answering_records_the_outcome_for_this_browser(self):
        self._answer(self.client, 'Invasion of Poland')

        attempt = QuestionAttempt.objects.get(question=self.question)
        self.assertTrue(attempt.is_correct)
        self.assertEqual(attempt.browser_session_key, self.client.session.session_key)

    def test_re_answering_updates_the_same_row(self):
        self._answer(self.client, 'Fall of France')
        self._answer(self.client, 'Invasion of Poland')

        attempts = QuestionAttempt.objects.filter(question=self.question)
        self.assertEqual(attempts.count(), 1)
        self.assertTrue(attempts.first().is_correct)

    def test_each_browser_keeps_its_own_outcome(self):
        self._answer(self.client, 'Invasion of Poland')
        self._answer(Client(), 'Fall of France')

        outcomes = set(
            QuestionAttempt.objects.values_list('browser_session_key', 'is_correct')
        )
        self.assertEqual(len(outcomes), 2)
        self.assertEqual({correct for _, correct in outcomes}, {True, False})


class SubjectAndLessonCreationTests(TestCase):
    def setUp(self):
        self.subject = Subject.objects.create(id='sub_gen', title='Biology')
        self.url = reverse('other_quiz:subject_detail', kwargs={'subject_id': self.subject.id})

    def _upload(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        return SimpleUploadedFile('notes.txt', 'Cells are the unit of life.'.encode(), 'text/plain')

    def test_overlong_subject_fields_are_trimmed_to_model_limits(self):
        # MySQL strict mode raised a 500 on an icon longer than max_length=10.
        self.client.post(reverse('other_quiz:create_subject'), {
            'title': 'T' * 300, 'icon': '📚' * 20, 'description': 'd',
        })
        created = Subject.objects.get(title__startswith='TTT')
        self.assertEqual(len(created.title), 200)
        self.assertEqual(len(created.icon), 10)

    @patch.dict('os.environ', {'GROQ_API_KEY': ''})
    @patch('apps.other_quiz.services.generate_ai_response')
    def test_lesson_uses_shared_ai_chain_without_groq_key(self, mock_ai):
        # Regression: lesson creation hard-required GROQ_API_KEY, so a
        # Gemini- or DeepSeek-only setup could never create a lesson.
        mock_ai.return_value = {
            'chapter': 'Chapter 1 ' + 'x' * 80,
            'title': 'Cells',
            'questions': [
                {'title': 'Unit', 'prompt': 'What is the unit of life?', 'q_type': 'MULTIPLE_CHOICE',
                 'options': ['A. Cell', 'B. Atom'], 'correct_answer': 'A. Cell', 'explanation': 'Cells.'},
                {'title': 'Broken', 'prompt': 'Answer not among options', 'q_type': 'MULTIPLE_CHOICE',
                 'options': ['A', 'B'], 'correct_answer': 'C'},
                {'title': 'Explain', 'prompt': 'Explain cells.', 'q_type': 'RUBRIC',
                 'correct_answer': 'Cells are the unit of life.', 'rubric_keywords': ['cell', 'life']},
            ],
        }

        response = self.client.post(self.url, {'document': self._upload()})

        self.assertRedirects(response, self.url)
        lesson = Lesson.objects.get(subject=self.subject)
        self.assertEqual(len(lesson.chapter), 50)
        self.assertEqual(
            list(lesson.questions.order_by('id').values_list('title', flat=True)),
            ['Unit', 'Explain'],
        )

    @patch.dict('os.environ', {'GROQ_API_KEY': ''})
    def test_no_ai_available_shows_one_error_message(self):
        response = self.client.post(self.url, {'document': self._upload()}, follow=True)

        self.assertFalse(Lesson.objects.filter(subject=self.subject).exists())
        # base.html already renders messages; the page must not repeat them.
        self.assertContains(response, 'No AI provider is available', count=1)


class RubricSafetyTests(SimpleTestCase):
    def test_open_response_without_reference_or_rubric_is_not_evaluated(self):
        question = Question(correct_answer='', rubric_keywords=[])
        self.assertFalse(can_evaluate_open_response(question))

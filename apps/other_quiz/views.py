import logging
import uuid

from django.contrib import messages
from django.shortcuts import render, get_object_or_404, redirect

from .models import Lesson, QuestionAttempt, Subject
from .services import (
    evaluate_multiple_choice,
    evaluate_rubric,
    generate_and_save_lesson
)

logger = logging.getLogger(__name__)


def home(request):
    subjects = Subject.objects.all()
    return render(request, 'other_quiz/home.html', {'subjects': subjects})

def create_subject(request):
    if request.method != "POST":
        return redirect('other_quiz:home')
    # Trimmed to the model's max_length: MySQL strict mode raises (a 500)
    # on longer values instead of truncating, SQLite silently stores them.
    title = (request.POST.get("title") or "").strip()[:200]
    icon = (request.POST.get("icon") or "").strip()[:10] or "📖"
    description = (request.POST.get("description") or "").strip()

    if title:
        sub_id = f"sub_{uuid.uuid4().hex[:8]}"  # Auto-generate unique ID
        Subject.objects.create(
            id=sub_id,
            title=title,
            icon=icon,
            description=description
        )
        messages.success(request, f"Created “{title}”.")
    else:
        messages.error(request, "Enter a subject name.")

    return redirect('other_quiz:home')

def subject_detail(request, subject_id):
    subject = get_object_or_404(Subject, id=subject_id)

    if request.method == "POST":
        if not request.FILES.get("document"):
            messages.error(request, "Choose a file to create a lesson from.")
        else:
            try:
                generate_and_save_lesson(subject, request.FILES["document"])
                messages.success(request, "🎉 New lesson created!")
            except ValueError as e:
                messages.error(request, f"Could not create the lesson: {e}")
            except Exception:
                logger.exception("Lesson generation failed for subject %s", subject.id)
                messages.error(request, "Could not create the lesson because of an unexpected error. Please try again.")

        return redirect('other_quiz:subject_detail', subject_id=subject.id)

    return render(request, 'other_quiz/subject_detail.html', {
        'subject': subject,
        'lessons': subject.lessons.all(),
    })

def _record_attempt(request, question):
    """Save this browser's own outcome next to the shared Question.is_correct.

    Written on every submission so re-answering updates the row rather than
    piling up duplicates — the Progress page only cares where the learner
    stands now. A session key has to exist first: Django does not assign one
    until something is stored in the session, and anonymous visitors here
    never store anything.
    """
    if request.session.session_key is None:
        request.session.create()
    QuestionAttempt.objects.update_or_create(
        question=question,
        browser_session_key=request.session.session_key,
        defaults={'is_correct': question.is_correct},
    )

def lesson_detail(request, subject_id, lesson_id):
    subject = get_object_or_404(Subject, id=subject_id)
    lesson = get_object_or_404(Lesson, id=lesson_id, subject=subject)
    questions = list(lesson.questions.order_by('id'))

    if request.method == "POST":
        question_id = request.POST.get("question_id")
        action = request.POST.get("action")
        question = next((q for q in questions if str(q.id) == str(question_id)), None)

        if question:
            if action == "submit_mc":
                selected = request.POST.get(f"question_{question_id}")
                question.eval = evaluate_multiple_choice(selected, question.correct_answer)
                question.is_correct = question.eval['is_correct']
                question.save()
                _record_attempt(request, question)

            elif action == "submit_rubric":
                user_text = request.POST.get("rubric_text", "")
                question.eval = evaluate_rubric(user_text, question.rubric_keywords or [])
                question.is_correct = question.eval['is_passed']
                question.save()
                _record_attempt(request, question)

    return render(request, 'other_quiz/lesson_detail.html', {
        'subject': subject,
        'lesson': lesson,
        'questions': questions
    })

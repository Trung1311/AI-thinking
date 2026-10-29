import os
import json
import logging
import uuid
from pypdf import PdfReader
import docx
from groq import Groq
from django.db import transaction

from apps.ai_engine.client import generate_ai_response
from apps.ai_engine.exceptions import AIEngineError

from .models import Subject, Lesson, Question
from .ai_prompts import LESSON_GENERATION_PROMPT

logger = logging.getLogger(__name__)

# Real JSON Schema for the shared provider chain (Gemini rejects anything else).
LESSON_SCHEMA = {
    'type': 'object',
    'properties': {
        'chapter': {'type': 'string'},
        'title': {'type': 'string'},
        'questions': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'title': {'type': 'string'},
                    'prompt': {'type': 'string'},
                    'q_type': {'type': 'string', 'enum': ['MULTIPLE_CHOICE', 'RUBRIC']},
                    'options': {'type': 'array', 'items': {'type': 'string'}},
                    'correct_answer': {'type': 'string'},
                    'rubric_keywords': {'type': 'array', 'items': {'type': 'string'}},
                    'explanation': {'type': 'string'},
                },
                'required': ['title', 'prompt', 'q_type', 'correct_answer'],
            },
        },
    },
    'required': ['chapter', 'title', 'questions'],
}


# --- 1. HÀM CHẤM ĐIỂM ---

def evaluate_multiple_choice(selected_option, correct_option):
    return {
        'is_correct': str(selected_option).strip() == str(correct_option).strip(),
        'selected': selected_option
    }

def evaluate_rubric(user_text, keywords, threshold=0.75):
    if not user_text or not keywords:
        return {'is_passed': False, 'score_percent': 0, 'user_text': user_text or ''}
        
    user_text_lower = user_text.lower()
    # So sánh trực tiếp chuỗi giúp tương thích tốt với tiếng Việt có dấu và tiếng Nhật/Trung
    matched_count = sum(1 for kw in keywords if kw.lower() in user_text_lower)
    total_keywords = len(keywords)
    
    score_percent = (matched_count / total_keywords) if total_keywords > 0 else 0
    
    return {
        'is_passed': score_percent >= threshold,
        'score_percent': round(score_percent * 100),
        'user_text': user_text
    }


# --- 2. HÀM TRÍCH XUẤT VĂN BẢN TỪ FILE ---

def extract_text_from_file(uploaded_file) -> str:
    """Trích xuất chuỗi văn bản từ file (PDF, DOCX, TXT) với kiểm tra kích thước & đa bảng mã."""
    # Giới hạn kích thước file tối đa 10MB
    if uploaded_file.size > 10 * 1024 * 1024:
        raise ValueError("The file is too large (10 MB maximum).")

    uploaded_file.seek(0)
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    text = ""

    try:
        if ext == ".pdf":
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

        elif ext == ".docx":
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs if p.text])

        elif ext == ".txt":
            content = uploaded_file.read()
            # Thử decode với nhiều bảng mã phổ biến (UTF-8, Shift-JIS tiếng Nhật, v.v.)
            for encoding in ['utf-8', 'shift_jis', 'euc-jp', 'utf-16']:
                try:
                    text = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if not text:
                text = content.decode('utf-8', errors='ignore')

    except ValueError:
        raise
    except Exception:
        logger.warning("Could not read uploaded file %s", uploaded_file.name, exc_info=True)
        return ""

    return text.strip()


# --- 3. HÀM GỌI GROQ API & LƯU DATABASE ---

def can_evaluate_open_response(question):
    """Only evaluate an open response when local reference evidence exists."""
    return bool(
        str(getattr(question, 'correct_answer', '') or '').strip()
        or getattr(question, 'rubric_keywords', None)
    )


def _request_lesson_data(raw_text):
    """Ask AI for the lesson JSON.

    Groq's gpt-oss-120b stays the first choice when GROQ_API_KEY is set; if
    Groq is not configured or fails, the shared provider chain (DeepSeek ->
    Gemini -> Groq, from whichever keys exist) is tried — before, a setup
    with only a Gemini or DeepSeek key could never create a lesson.
    """
    prompt = LESSON_GENERATION_PROMPT.format(raw_text=raw_text[:10000])

    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        try:
            response = Groq(api_key=api_key).chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            logger.warning("Groq lesson generation failed; trying the shared AI chain", exc_info=True)

    try:
        return generate_ai_response(
            system_prompt="You create review lessons from study material and answer only with JSON.",
            user_prompt=prompt,
            response_schema=LESSON_SCHEMA,
        )
    except AIEngineError as exc:
        raise ValueError(
            "No AI provider is available right now. Configure GROQ_API_KEY, "
            "GEMINI_API_KEY or DEEPSEEK_API_KEY, or try again later."
        ) from exc


def _clean_text(value, max_length=None):
    text = str(value or "").strip()
    return text[:max_length] if max_length else text


def _valid_questions(raw_questions):
    """Keep only questions the lesson page can actually grade."""
    questions = []
    for q in raw_questions if isinstance(raw_questions, list) else []:
        if not isinstance(q, dict) or not _clean_text(q.get("prompt")):
            continue
        q_type = q.get("q_type") if q.get("q_type") in {"MULTIPLE_CHOICE", "RUBRIC"} else "MULTIPLE_CHOICE"
        correct_answer = _clean_text(q.get("correct_answer"))
        options = None
        rubric_keywords = None
        if q_type == "MULTIPLE_CHOICE":
            options = [_clean_text(o) for o in q.get("options") or [] if _clean_text(o)]
            # Without the answer among >= 2 options, no submission could be marked correct.
            if len(options) < 2 or correct_answer not in options:
                continue
        else:
            rubric_keywords = [_clean_text(k) for k in q.get("rubric_keywords") or [] if _clean_text(k)]
            if not rubric_keywords:
                continue
        questions.append({
            "title": _clean_text(q.get("title"), 200) or "Question",
            "prompt": _clean_text(q.get("prompt")),
            "q_type": q_type,
            "options": options,
            "correct_answer": correct_answer,
            "rubric_keywords": rubric_keywords,
            "explanation": _clean_text(q.get("explanation")),
        })
    return questions


def generate_and_save_lesson(subject: Subject, uploaded_file) -> Lesson:
    # 1. Đọc văn bản từ file
    raw_text = extract_text_from_file(uploaded_file)
    if not raw_text:
        raise ValueError("Could not read any text from this file, or the file is empty.")

    # 2. Gọi AI (Groq trước, sau đó chuỗi provider dùng chung)
    ai_data = _request_lesson_data(raw_text)
    if not isinstance(ai_data, dict):
        raise ValueError("The AI returned an unexpected response. Please try again.")
    questions = _valid_questions(ai_data.get("questions"))
    if not questions:
        raise ValueError("The AI did not return any usable questions. Please try again.")

    # 3. Lưu vào Database
    with transaction.atomic():
        lesson = Lesson.objects.create(
            id=f"les_{uuid.uuid4().hex[:8]}",
            subject=subject,
            # CharField limits: MySQL strict mode rejects longer AI-written values.
            chapter=_clean_text(ai_data.get("chapter"), 50) or "Chapter 1",
            title=_clean_text(ai_data.get("title"), 200) or "New lesson",
        )
        for q in questions:
            Question.objects.create(lesson=lesson, **q)

    return lesson

"""
Бот-помощник: База знаний по Обществознанию
Работает ТОЛЬКО по загруженным файлам, не ищет в интернете.
Всегда указывает источник ответа.
"""

import os
import logging
from pathlib import Path
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document

# ─── Настройки ────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("SOCIAL_KB_BOT_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
FILES_DIR = Path("files/social")      # папка с учебными материалами
MAX_HISTORY = 6                        # сколько сообщений помним в диалоге
MAX_CONTEXT_CHARS = 120_000           # сколько символов из файлов передаём в запрос

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ─── Глобальное хранилище ──────────────────────────────────────
knowledge_base: list[dict] = []   # [{"name": "файл.pdf", "text": "...", "pages": {...}}]
chat_histories: dict[int, list] = {}   # chat_id → список сообщений

# ─── Загрузка файлов ──────────────────────────────────────────

def load_pdf(path: Path) -> tuple[str, dict]:
    """Читает PDF через pdfminer, возвращает (полный_текст, {})."""
    text = pdf_extract_text(str(path)) or ""
    return text.strip(), {}


def load_docx(path: Path) -> tuple[str, dict]:
    """Читает DOCX, страницы не нумерованы — возвращает весь текст."""
    doc = Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return text, {}


def load_txt(path: Path) -> tuple[str, dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text, {}


def load_all_files() -> list[dict]:
    """Загружает все файлы из FILES_DIR при старте бота."""
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    for f in FILES_DIR.iterdir():
        if not f.is_file():
            continue
        try:
            ext = f.suffix.lower()
            if ext == ".pdf":
                text, pages = load_pdf(f)
            elif ext == ".docx":
                text, pages = load_docx(f)
            elif ext in (".txt", ".md"):
                text, pages = load_txt(f)
            else:
                log.info("Пропущен файл: %s (неизвестный тип)", f.name)
                continue
            result.append({"name": f.name, "text": text, "pages": pages})
            log.info("Загружен файл: %s (%d символов)", f.name, len(text))
        except Exception as e:
            log.error("Ошибка при загрузке %s: %s", f.name, e)
    return result


# ─── Поиск по ключевым словам ─────────────────────────────────

def search_relevant_chunks(query: str, chunk_size: int = 1500, top_n: int = 20) -> str:
    """Ищет релевантные куски текста по ключевым словам из вопроса."""
    if not knowledge_base:
        return "Файлы не загружены."

    # Извлекаем ключевые слова (слова длиннее 3 букв)
    keywords = [w.lower() for w in query.split() if len(w) > 3]
    if not keywords:
        keywords = query.lower().split()

    scored_chunks = []
    for doc in knowledge_base:
        text = doc["text"]
        if not text:
            continue
        # Разбиваем файл на куски
        for i in range(0, len(text), chunk_size // 2):
            chunk = text[i:i + chunk_size]
            chunk_lower = chunk.lower()
            # Считаем совпадения ключевых слов
            score = sum(chunk_lower.count(kw) for kw in keywords)
            if score > 0:
                scored_chunks.append((score, doc["name"], chunk))

    if not scored_chunks:
        # Если ничего не нашли — берём начало каждого файла
        parts = []
        for doc in knowledge_base:
            if doc["text"]:
                parts.append(f"=== {doc['name']} ===\n{doc['text'][:500]}")
        return "\n\n".join(parts[:10])

    # Сортируем по релевантности и берём лучшие куски
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    result_parts = []
    total = 0
    for score, name, chunk in scored_chunks[:top_n]:
        block = f"=== Файл: {name} ===\n{chunk}"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        result_parts.append(block)
        total += len(block)

    return "\n\n".join(result_parts)


def build_system_prompt(query: str = "") -> str:
    file_names = ", ".join(d["name"] for d in knowledge_base) if knowledge_base else "нет файлов"
    context = search_relevant_chunks(query) if query else search_relevant_chunks("право закон государство")
    return f"""Ты — ИИ-помощник по предмету Обществознание для подготовки к ЕГЭ.

СТРОГИЕ ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе материалов ниже. Не используй знания из интернета или другие источники.
2. После каждого ответа обязательно укажи источник: имя файла.
3. Если ответа в материалах нет — честно скажи: «В загруженных материалах эта тема не найдена».
4. Отвечай по-русски, чётко и по делу.
5. Для вопросов ЕГЭ — давай структурированный ответ с примерами из материалов.

ЗАГРУЖЕННЫЕ ФАЙЛЫ: {file_names}

━━━ НАЙДЕННЫЕ МАТЕРИАЛЫ ПО ЗАПРОСУ ━━━
{context}
━━━ КОНЕЦ МАТЕРИАЛОВ ━━━"""


# ─── OpenAI GPT ───────────────────────────────────────────────

def ask_gemini(user_message: str, history: list) -> str:
    """Отправляет запрос в OpenAI GPT через REST API."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [{"role": "system", "content": build_system_prompt(user_message)}]
    for msg in history[:-1]:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.3
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ─── Telegram handlers ────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    chat_histories[chat_id] = []

    if knowledge_base:
        files_list = "\n".join(f"  📄 {d['name']}" for d in knowledge_base)
        text = (
            "📚 *Бот по Обществознанию* запущен!\n\n"
            f"Загружено файлов: *{len(knowledge_base)}*\n{files_list}\n\n"
            "Задавай вопросы — отвечу только по этим материалам и укажу источник.\n\n"
            "Команды:\n"
            "/files — список файлов\n"
            "/reset — начать диалог заново"
        )
    else:
        text = (
            "⚠️ *Файлы не загружены!*\n\n"
            f"Добавь PDF/DOCX/TXT файлы в папку `{FILES_DIR}` и перезапусти бота."
        )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_files(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not knowledge_base:
        await update.message.reply_text(f"Папка `{FILES_DIR}` пуста. Добавь файлы и перезапусти бота.")
        return
    lines = [f"📁 *Загруженные материалы:*\n"]
    for d in knowledge_base:
        chars = len(d["text"])
        pages = len(d["pages"])
        info = f"{pages} стр." if pages else f"{chars:,} символов"
        lines.append(f"• {d['name']} — {info}")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    chat_histories[chat_id] = []
    await update.message.reply_text("🔄 Память диалога очищена. Начинаем заново!")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text.strip()

    if not knowledge_base:
        await update.message.reply_text(
            f"⚠️ Файлы не загружены. Добавь материалы в папку `{FILES_DIR}` и перезапусти бота."
        )
        return

    # Инициализируем историю если нет
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []

    # Добавляем вопрос в историю
    chat_histories[chat_id].append({"role": "user", "content": user_text})

    # Обрезаем историю
    if len(chat_histories[chat_id]) > MAX_HISTORY:
        chat_histories[chat_id] = chat_histories[chat_id][-MAX_HISTORY:]

    # Индикатор печатания
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        answer = ask_gemini(user_text, chat_histories[chat_id])
        chat_histories[chat_id].append({"role": "model", "content": answer})
        await update.message.reply_text(answer)
    except Exception as e:
        log.error("Ошибка Gemini: %s", e)
        await update.message.reply_text(
            "❌ Ошибка при обращении к ИИ. Проверь GEMINI_API_KEY или попробуй позже."
        )


# ─── Запуск ───────────────────────────────────────────────────

def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("Не задана переменная SOCIAL_KB_BOT_TOKEN")
    if not OPENAI_API_KEY:
        raise ValueError("Не задана переменная OPENAI_API_KEY")

    global knowledge_base
    log.info("Загружаю файлы из %s ...", FILES_DIR)
    knowledge_base = load_all_files()
    log.info("Загружено файлов: %d", len(knowledge_base))

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("files", cmd_files))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("Бот Обществознание (База знаний) запущен.")
    app.run_polling()


if __name__ == "__main__":
    main()

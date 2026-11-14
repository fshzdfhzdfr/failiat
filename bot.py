import logging
import os
import re
from collections import defaultdict, deque
from time import monotonic

from openai import AsyncOpenAI
from telegram import Update
from telegram.constants import ChatType
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("pixel-game-bot")

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
FREEMODEL_API_KEY = os.environ["FREEMODEL_API_KEY"]
ALLOWED_CHAT_USERNAME = os.getenv("ALLOWED_CHAT_USERNAME", "Pixel_Mediair").lstrip("@").lower()
QUIET_MINUTES = int(os.getenv("QUIET_MINUTES", "15"))
MODEL = os.getenv("FREEMODEL_MODEL", "gemini-3.7-flash")
MAX_REPLY_CHARS = int(os.getenv("MAX_REPLY_CHARS", "900"))
REPLY_COOLDOWN_SECONDS = int(os.getenv("REPLY_COOLDOWN_SECONDS", "20"))

ai = AsyncOpenAI(api_key=FREEMODEL_API_KEY, base_url="https://freemodel.in/v1")
# Conversation context is intentionally short and in-memory; Railway restarts clear it.
history = defaultdict(lambda: deque(maxlen=16))
last_reply_at = {}

SYSTEM_PROMPT = """تو Pixel، عضو خوش‌برخورد و مختصر گروه فارسی‌زبان بازی‌های رایانه‌ای هستی.
فقط درباره بازی‌های رایانه‌ای، بازی‌های ویدیویی، کنسول و سخت‌افزار/اخبار مرتبط گفتگو کن.
اگر پیام کاربر خارج از این موضوع است و سؤال مستقیمی از تو نپرسیده، دقیقاً فقط [SKIP] برگردان.
اگر پیام مرتبط است یا سؤال مستقیمی از تو دارد، پاسخ کوتاه، طبیعی و مفید به فارسی بده؛ از بحث‌های سیاسی، توهین، اطلاعات ساختگی و اسپم پرهیز کن.
در شروع بحث پس از سکوت، یک پرسش یا چالش اجتماعی درباره بازی‌ها مطرح کن که اعضا را به پاسخ‌دادن دعوت کند؛ سؤال تکراری نساز.
نام بازی، پلتفرم یا نسخه را در صورت نامشخص‌بودن حدس نزن. پاسخ را حداکثر در 3 جمله نگه دار.
"""


def is_target_group(update: Update) -> bool:
    chat = update.effective_chat
    return bool(
        chat
        and chat.type in (ChatType.GROUP, ChatType.SUPERGROUP)
        and (chat.username or "").lower() == ALLOWED_CHAT_USERNAME
    )


def clean_name(update: Update) -> str:
    user = update.effective_user
    if not user:
        return "عضو گروه"
    return user.first_name or user.username or "عضو گروه"


async def ask_model(chat_id: int, prompt: str, *, game_opener: bool = False) -> str | None:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if not game_opener:
        messages.extend(history[chat_id])
    messages.append({"role": "user", "content": prompt})
    try:
        result = await ai.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.85 if game_opener else 0.65,
            max_tokens=240,
        )
        answer = (result.choices[0].message.content or "").strip()
        if not answer or answer == "[SKIP]":
            return None
        answer = re.sub(r"^\[SKIP\].*$", "", answer, flags=re.DOTALL).strip()
        return answer[:MAX_REPLY_CHARS] or None
    except Exception:
        log.exception("FreeModel request failed")
        return None


async def quiet_time_prompt(context: ContextTypes.DEFAULT_TYPE) -> None:
    job = context.job
    chat_id = job.chat_id
    # A replacement job is scheduled on every human message. This identity check
    # avoids a stale timer sending a prompt if two updates arrive nearly together.
    if context.application.bot_data.get(f"quiet_job_{chat_id}") != job.data:
        return
    starter = await ask_model(
        chat_id,
        "اکنون در گروه ۱۵ دقیقه سکوت بوده است. یک بحث یا چالش گیمی کوتاه و تازه به فارسی آغاز کن.",
        game_opener=True,
    )
    if starter:
        try:
            await context.bot.send_message(chat_id=chat_id, text=starter)
            history[chat_id].append({"role": "assistant", "content": starter})
        except Exception:
            log.exception("Could not send quiet-time prompt")


async def schedule_quiet_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    key = f"quiet_job_{chat_id}"
    for old_job in context.job_queue.get_jobs_by_name(key):
        old_job.schedule_removal()
    marker = object()
    context.application.bot_data[key] = marker
    context.job_queue.run_once(
        quiet_time_prompt,
        when=QUIET_MINUTES * 60,
        chat_id=chat_id,
        name=key,
        data=marker,
    )


async def on_group_activity(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_target_group(update) or not update.effective_message:
        return
    if update.effective_user and update.effective_user.is_bot:
        return

    await schedule_quiet_prompt(update, context)
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip()
    if not text:
        return

    chat_id = update.effective_chat.id
    username = (context.bot.username or "").lower()
    addressed = bool(update.message.reply_to_message and update.message.reply_to_message.from_user
                     and update.message.reply_to_message.from_user.is_bot
                     and (update.message.reply_to_message.from_user.username or "").lower() == username)
    mentioned = bool(username and re.search(rf"@{re.escape(username)}\b", text, flags=re.IGNORECASE))

    now = monotonic()
    if not (addressed or mentioned) and now - last_reply_at.get(chat_id, 0) < REPLY_COOLDOWN_SECONDS:
        history[chat_id].append({"role": "user", "content": f"{clean_name(update)}: {text}"})
        return

    # Let the AI decide whether this is worth answering. It returns [SKIP] for
    # unrelated chatter, while direct mentions/replies explicitly ask it to answer.
    prefix = "پاسخ بده. " if addressed or mentioned else "اگر این پیام نیازمند پاسخ گیمی یا پاسخ مستقیم است، جواب بده؛ وگرنه [SKIP] برگردان. "
    prompt = f"{prefix}{clean_name(update)} نوشت: {text}"
    answer = await ask_model(chat_id, prompt)
    history[chat_id].append({"role": "user", "content": f"{clean_name(update)}: {text}"})
    if not answer:
        return

    last_reply_at[chat_id] = monotonic()
    history[chat_id].append({"role": "assistant", "content": answer})
    try:
        await update.message.reply_text(answer, disable_web_page_preview=True)
    except Exception:
        log.exception("Could not send AI reply")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message:
        await update.effective_message.reply_text(
            "من ربات گفت‌وگوی گیم هستم. در گروه مجاز، بعد از ۱۵ دقیقه سکوت بحثی درباره بازی‌های رایانه‌ای شروع می‌کنم و به گفتگوهای مرتبط پاسخ می‌دهم."
        )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if update.effective_message:
        if chat and chat.type in (ChatType.GROUP, ChatType.SUPERGROUP) and not is_target_group(update):
            await update.effective_message.reply_text("این ربات فقط برای گروه مجاز پیکربندی شده است.")
            return
        await update.effective_message.reply_text(
            f"وضعیت: فعال\nگروه مجاز: @{ALLOWED_CHAT_USERNAME}\nزمان سکوت: {QUIET_MINUTES} دقیقه\nمدل: {MODEL}"
        )


def main() -> None:
    if QUIET_MINUTES < 1:
        raise ValueError("QUIET_MINUTES must be at least 1")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(MessageHandler(filters.ALL, on_group_activity))
    log.info("Starting polling; allowed group @%s; quiet timer %s minutes", ALLOWED_CHAT_USERNAME, QUIET_MINUTES)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

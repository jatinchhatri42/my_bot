"""
test_bot.py — minimal bot, no Firebase, no tools
Just proves Telegram + Ollama are talking to each other.
Run: python test_bot.py
"""

import asyncio
import sys
import logging
import requests
from dotenv import load_dotenv
import os

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")


def ask_ollama(text: str) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model":   OLLAMA_MODEL,
            "messages": [{"role": "user", "content": text}],
            "stream":  False,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


async def handle(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_msg = update.message.text or ""
    print(f">> User: {user_msg}")
    await update.message.reply_text("⏳ Thinking...")
    try:
        reply = await asyncio.to_thread(ask_ollama, user_msg)
        print(f">> Ollama: {reply[:80]}")
        await update.message.reply_text(reply)
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")
        print(f"ERROR: {e}")


def main():
    print(f"Starting test bot — model: {OLLAMA_MODEL}")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT, handle))
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()

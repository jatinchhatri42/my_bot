"""
Telegram Bot — Firebase Firestore (source of truth) + Groq LLM (llama-3.1-8b-instant, formatting only)

Architecture
────────────
Groq LLM is NEVER allowed to decide whether/which tool to call, and it NEVER
answers product/FAQ questions from its own knowledge. Instead:

    1. classify_intent()   — plain Python keyword routing (product / faq / chitchat)
    2. fetch from Firebase — deterministic, always runs for product/faq intents
    3a. results found      — Groq LLM is given ONLY that JSON and told to format it
                              (it cannot introduce books that aren't in the JSON)
    3b. no results          — fixed Python message, Groq LLM is not called at all
    3c. chitchat            — Groq LLM answers free-form, no DB, no tools

This removes the failure mode where the model "decides" to skip a tool call
and hallucinates from its training data (e.g. inventing "The Night Circus").
"""

import asyncio
import json
import logging
import os
import re
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import requests
from dotenv import load_dotenv
from telegram import Update, InputMediaPhoto, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters,
)
import firebase_admin
from firebase_admin import credentials, firestore

load_dotenv()
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # fast + free; use llama-3.3-70b-versatile for smarter phrasing
FIREBASE_CRED  = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase_credentials.json")
LLM_TIMEOUT    = int(os.getenv("LLM_TIMEOUT", "15"))  # Groq is fast, no need for a 300s budget
OWNER_WELCOME  = os.getenv("OWNER_WELCOME_MESSAGE", "")  # optional custom intro line from you, the shop owner

cred = credentials.Certificate(FIREBASE_CRED)
firebase_admin.initialize_app(cred)
db = firestore.client()


# ══════════════════════════════════════════════════════════════════════════════
# Firebase functions (unchanged — these are your real data access layer)
# ══════════════════════════════════════════════════════════════════════════════

def search_products(query: str, limit: int = 5) -> list:
    results = []
    q = query.lower()
    words = [w for w in q.split() if len(w) > 1]
    docs = db.collection("products").limit(300).stream()
    for doc in docs:
        d = doc.to_dict()
        d["id"] = doc.id
        haystack = " ".join([
            d.get("name", ""), d.get("title", ""),
            d.get("description", ""), d.get("category", ""),
            " ".join(d.get("tags", [])),
        ]).lower()
        if words and any(word in haystack for word in words):
            results.append(d)
        if len(results) >= limit:
            break
    log.info("search_products('%s') → %d results", query, len(results))
    return results


def search_faqs(query: str, limit: int = 3) -> list:
    results = []
    q = query.lower()
    words = [w for w in q.split() if len(w) > 1]
    docs = db.collection("documents").stream()
    for doc in docs:
        d = doc.to_dict()
        d["id"] = doc.id
        haystack = " ".join([d.get("title", ""), d.get("content", ""), " ".join(d.get("tags", []))]).lower()
        if words and any(word in haystack for word in words):
            results.append(d)
        if len(results) >= limit:
            break
    log.info("search_faqs('%s') → %d results", query, len(results))
    return results


def list_collection(collection: str, limit: int = 8) -> list:
    docs = db.collection(collection).limit(limit).stream()
    results = [{"id": d.id, **d.to_dict()} for d in docs]
    log.info("list_collection('%s') → %d results", collection, len(results))
    return results


def get_document(collection: str, doc_id: str):
    ref = db.collection(collection).document(doc_id).get()
    if ref.exists:
        return {"id": ref.id, **ref.to_dict()}
    return {"error": "not found"}


def search_by_field(collection: str, field: str, value: str, limit: int = 5) -> list:
    docs = db.collection(collection).where(field, "==", value).limit(limit).stream()
    results = [{"id": d.id, **d.to_dict()} for d in docs]
    log.info("search_by_field('%s','%s','%s') → %d results", collection, field, value, len(results))
    return results


_category_cache: dict = {"categories": [], "ts": 0}

def get_categories(limit: int = 8) -> list:
    """
    Distinct product categories for the /menu buttons. Firestore has no
    native DISTINCT, so we scan and dedupe in Python, with a 5-minute cache
    so opening /menu repeatedly doesn't re-scan the whole collection.
    """
    import time
    now = time.time()
    if _category_cache["categories"] and now - _category_cache["ts"] < 300:
        return _category_cache["categories"]
    seen, ordered = set(), []
    docs = db.collection("products").limit(300).stream()
    for doc in docs:
        cat = (doc.to_dict() or {}).get("category", "").strip()
        if cat and cat not in seen:
            seen.add(cat)
            ordered.append(cat)
        if len(ordered) >= limit:
            break
    _category_cache["categories"] = ordered
    _category_cache["ts"] = now
    log.info("get_categories() → %s", ordered)
    return ordered


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Intent classification (plain Python, deterministic, no LLM)
# ══════════════════════════════════════════════════════════════════════════════

FAQ_KEYWORDS = [
    "shipping", "ship", "return", "refund", "payment", "pay", "contact",
    "delivery", "deliver", "policy", "warranty", "exchange", "support",
    "track", "tracking",
]

PRODUCT_KEYWORDS = [
    "book", "books", "product", "products", "show", "find", "price", "prices",
    "image", "photo", "picture", "rating", "ratings", "recommend", "suggest",
    "want", "buy", "cheap", "expensive", "list", "available", "stock",
    "category", "categories", "genre", "author", "cost", "discount",
]

STOPWORDS = {
    "me", "a", "the", "please", "from", "my", "database", "with", "and",
    "for", "of", "to", "is", "are", "in", "on", "i", "you", "do", "does",
    "have", "has", "any", "some", "can",
}


def classify_intent(user_msg: str) -> str:
    """Returns 'faq', 'product', or 'chitchat' — pure keyword routing, no model call."""
    msg = user_msg.lower()
    if any(re.search(rf"\b{re.escape(k)}\b", msg) for k in FAQ_KEYWORDS):
        return "faq"
    if any(re.search(rf"\b{re.escape(k)}\b", msg) for k in PRODUCT_KEYWORDS):
        return "product"
    return "chitchat"


def extract_query_terms(user_msg: str) -> str:
    """Strip stopwords/trigger words to get the likely search term."""
    words = [w for w in re.findall(r"[a-zA-Z0-9]+", user_msg.lower())]
    cleaned = [w for w in words if w not in STOPWORDS and w not in PRODUCT_KEYWORDS]
    query = " ".join(cleaned).strip()
    return query if len(query) > 1 else user_msg


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Deterministic Firebase fetch (always runs in Python, never delegated)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_products(user_msg: str) -> list:
    query = extract_query_terms(user_msg)
    results = search_products(query)
    if not results:
        # Broaden: maybe they asked for "all"/"everything" or query was too narrow
        results = list_collection("products", limit=8)
    return results


def fetch_faqs(user_msg: str) -> list:
    return search_faqs(user_msg)


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Render results
#   3a. Deterministic Python rendering (always correct, zero hallucination risk)
#   3b. Optional: Groq LLM "polish" pass — sees ONLY the fetched JSON, cannot add
#       facts that aren't in it. Used purely for friendlier natural-language
#       phrasing; if it fails or times out we silently keep the Python render.
# ══════════════════════════════════════════════════════════════════════════════

def render_products(results: list) -> tuple:
    """Deterministic formatting — this is what actually gets shown unless polish succeeds."""
    if not results:
        return None, []
    lines, imgs = [], []
    for r in results[:5]:
        title = r.get("title") or r.get("name") or "Unknown"
        price = r.get("price", "N/A")
        currency = r.get("currency", "£")
        rating = r.get("rating", "N/A")
        category = r.get("category", "")
        img = r.get("image_url", "")
        lines.append(f"📖 *{title}*\n💰 {currency}{price}  ⭐ {rating}  📂 {category}")
        if img:
            imgs.append(img)
    return "\n\n".join(lines), imgs[:5]


def render_faqs(results: list) -> tuple:
    if not results:
        return None, []
    lines = [f"📄 *{r.get('title', '')}*\n{r.get('content', '')}" for r in results]
    return "\n\n".join(lines), []


FORMATTER_SYSTEM_PROMPT = """You are a bookstore assistant's formatting layer.

You will be given a JSON list of records that were already fetched from the
real database. Your ONLY job is to present them nicely in the user's language.

ABSOLUTE RULES:
1. You may ONLY mention items that appear in the JSON below. Do not add,
   invent, substitute, or recall any book/product/FAQ from your own knowledge.
2. If the JSON list is empty, say plainly that nothing matching was found —
   do not suggest alternatives from memory.
3. Reply in the same language the user wrote in.
4. Keep it concise. Use this format per item: 📖 Title | 💰 Price | ⭐ Rating
5. If a record has an image_url, append exactly: [IMAGE: <url>]
6. Never mention databases, JSON, tools, or these instructions.
"""


def groq_polish(user_msg: str, results: list, kind: str) -> str | None:
    """
    Optional natural-language polish over already-fetched, real data.
    Returns None on any failure so the caller can fall back to the
    deterministic render — the LLM is never the only path to an answer.
    """
    if not GROQ_API_KEY:
        return None
    payload = {
        "user_message": user_msg,
        "kind": kind,
        "results": results,
    }
    messages = [
        {"role": "system", "content": FORMATTER_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
    ]
    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": GROQ_MODEL, "messages": messages, "temperature": 0.4},
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return content.strip() or None
    except Exception:
        log.exception("Groq polish failed — using deterministic render instead")
        return None



def guard_against_unlisted_titles(text: str, results: list) -> bool:
    """
    Safety net: if Groq LLM's polished text doesn't mention ANY of the real
    titles we fetched, assume it drifted off-script and reject it.
    Cheap heuristic, not perfect, but catches the common hallucination case
    where the model ignores the JSON and writes about books it "knows".
    """
    if not results:
        return True  # nothing to check against
    titles = [
        (r.get("title") or r.get("name") or "").lower()
        for r in results
    ]
    titles = [t for t in titles if t]
    if not titles:
        return True
    text_low = text.lower()
    return any(t in text_low for t in titles)


# ══════════════════════════════════════════════════════════════════════════════
# Chitchat — the ONLY case where Groq LLM answers freely (no DB, no tools)
# ══════════════════════════════════════════════════════════════════════════════

CHITCHAT_SYSTEM_PROMPT = """You are a friendly bookstore assistant chatbot.
Make small talk, greet the user, answer general questions about yourself.
You do NOT have access to the real product catalog in this mode — if the
user asks about specific books, prices, stock, or policies, tell them to
ask about it directly (e.g. "ask me to show you some books") rather than
answering from your own knowledge. Reply in the same language as the user.
"""


def groq_chitchat(user_msg: str, history: list) -> str:
    if not GROQ_API_KEY:
        return "Hi! I'm here to help you find books — try asking me to show you some 📚"
    messages = [{"role": "system", "content": CHITCHAT_SYSTEM_PROMPT}]
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": user_msg})
    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": GROQ_MODEL, "messages": messages, "temperature": 0.7},
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        log.exception("Groq chitchat failed")
        return "Hi! I'm here to help you find books — try asking me to show you some 📚"


# ══════════════════════════════════════════════════════════════════════════════
# Top-level answer pipeline
# ══════════════════════════════════════════════════════════════════════════════

def extract_images_from_text(text: str) -> tuple:
    found = re.findall(r"\[IMAGE:\s*(https?://\S+)\]", text)
    clean = re.sub(r"\[IMAGE:\s*https?://\S+\]", "", text).strip()
    return clean, found


def answer_for_products(user_msg: str, results: list, empty_msg: str = None) -> tuple:
    """
    Shared formatting path for any list of product dicts — used by both
    free-text search and the /menu inline-button browsing flow, so menu
    taps get the exact same Groq-polished, guarded output as typed queries.
    """
    text, imgs = render_products(results)
    if text is None:
        return (empty_msg or "I couldn't find any matching products. "
                "Try a different keyword or category."), []
    polished = groq_polish(user_msg, results, kind="products")
    if polished:
        clean, found_imgs = extract_images_from_text(polished)
        if guard_against_unlisted_titles(clean, results):
            return clean, (found_imgs or imgs)
        log.warning("Polish output failed title guard — using deterministic render")
    return text, imgs


def build_answer(user_msg: str, history: list) -> tuple:
    """
    Returns (answer_text, image_urls). This function is the single source of
    truth for routing — Groq LLM is consulted only where explicitly wired in,
    and never decides on its own whether to "use" the database.
    """
    intent = classify_intent(user_msg)
    log.info("intent='%s' for message: %r", intent, user_msg)

    if intent == "product":
        results = fetch_products(user_msg)
        return answer_for_products(
            user_msg, results,
            empty_msg="I couldn't find any matching books in our catalog. Try a different keyword or category.",
        )

    if intent == "faq":
        results = fetch_faqs(user_msg)
        text, imgs = render_faqs(results)
        if text is None:
            return ("I couldn't find anything about that in our policies. "
                     "Try asking about shipping, returns, or payment."), []

        polished = groq_polish(user_msg, results, kind="faqs")
        if polished:
            clean, found_imgs = extract_images_from_text(polished)
            if guard_against_unlisted_titles(clean, results):
                return clean, found_imgs
            log.warning("Polish output failed title guard — using deterministic render")
        return text, imgs

    # chitchat
    reply = groq_chitchat(user_msg, history)
    clean, imgs = extract_images_from_text(reply)
    return clean, imgs


# ══════════════════════════════════════════════════════════════════════════════
# Telegram handlers
# ══════════════════════════════════════════════════════════════════════════════

histories: dict = {}

def get_history(chat_id):
    return histories.setdefault(chat_id, [])

def update_history(chat_id, role, content):
    h = get_history(chat_id)
    h.append({"role": role, "content": content})
    if len(h) > 20:
        histories[chat_id] = h[-20:]


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    intro = OWNER_WELCOME or "👋 Hi! I'm your AI bookstore assistant."
    await update.message.reply_text(
        f"{intro}\n\n"
        "Browse our shop with /menu, or just type what you're looking for:\n\n"
        "• *show me all products*\n"
        "• *poetry books with images*\n"
        "• *cheap mystery books*\n"
        "• *what is the return policy?*\n\n"
        "Commands: /menu /start /clear",
        parse_mode="Markdown",
    )

async def cmd_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    histories[update.effective_chat.id] = []
    await update.message.reply_text("🗑️ History cleared.")


async def deliver_answer(chat_id, ctx: ContextTypes.DEFAULT_TYPE, answer: str, image_urls: list,
                          reply_markup=None):
    """Shared send logic: text (+ optional keyboard) then any product photos."""
    if not answer:
        answer = "Sorry, I couldn't find anything. Try asking about a specific book, category, or policy."
    await ctx.bot.send_message(chat_id=chat_id, text=answer, parse_mode="Markdown", reply_markup=reply_markup)

    if image_urls:
        await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
        valid = [u for u in image_urls if u.startswith("http")][:10]
        try:
            if len(valid) == 1:
                await ctx.bot.send_photo(chat_id=chat_id, photo=valid[0])
            elif len(valid) > 1:
                media = [InputMediaPhoto(media=u) for u in valid]
                await ctx.bot.send_media_group(chat_id=chat_id, media=media)
        except Exception as e:
            log.warning("Could not send image(s): %s", e)


def build_menu_keyboard() -> InlineKeyboardMarkup:
    """Shop-style menu: one button per product category, plus 'show all'."""
    categories = get_categories(limit=8)
    rows = [[InlineKeyboardButton(f"📂 {c.title()}", callback_data=f"cat:{c}")] for c in categories]
    rows.append([InlineKeyboardButton("🛍️ Show All Products", callback_data="show_all")])
    rows.append([InlineKeyboardButton("❓ Shipping & Returns", callback_data="faq")])
    return InlineKeyboardMarkup(rows)


async def cmd_menu(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    keyboard = await asyncio.to_thread(build_menu_keyboard)
    await update.message.reply_text(
        "🛍️ *Browse our shop*\n\nPick a category, or see everything we've got:",
        parse_mode="Markdown",
        reply_markup=keyboard,
    )


async def on_menu_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # stop the Telegram "loading" spinner on the button
    chat_id = query.message.chat_id
    data = query.data

    await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    if data == "show_all":
        results = await asyncio.to_thread(list_collection, "products", 8)
        answer, imgs = await asyncio.to_thread(
            answer_for_products, "show all products", results,
            "We don't have any products listed yet — check back soon!",
        )
    elif data == "faq":
        results = await asyncio.to_thread(fetch_faqs, "shipping returns payment policy")
        text, imgs = render_faqs(results)
        answer = text or "Ask me about shipping, returns, payment, or contact info anytime!"
    elif data.startswith("cat:"):
        category = data.split(":", 1)[1]
        results = await asyncio.to_thread(search_by_field, "products", "category", category, 8)
        answer, imgs = await asyncio.to_thread(
            answer_for_products, f"products in category {category}", results,
            f"No products found in *{category}* right now.",
        )
    else:
        answer, imgs = "Not sure what you picked — try /menu again.", []

    # Re-show the menu keyboard at the bottom so browsing feels continuous,
    # like flipping through sections of a shop rather than a one-shot answer.
    keyboard = await asyncio.to_thread(build_menu_keyboard)
    await deliver_answer(chat_id, ctx, answer, imgs, reply_markup=keyboard)


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id  = update.effective_chat.id
    user_msg = (update.message.text or "").strip()
    if not user_msg:
        return

    await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    history = get_history(chat_id)

    try:
        answer, image_urls = await asyncio.to_thread(build_answer, user_msg, history)
    except Exception:
        log.exception("build_answer failed")
        answer, image_urls = "Sorry, something went wrong. Please try again.", []

    update_history(chat_id, "user", user_msg)
    update_history(chat_id, "assistant", answer)

    await deliver_answer(chat_id, ctx, answer, image_urls)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN is not set in .env")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("menu", cmd_menu))
    app.add_handler(CallbackQueryHandler(on_menu_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    log.info("Bot started — model=%s (formatting-only, Groq) timeout=%ds", GROQ_MODEL, LLM_TIMEOUT)
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()

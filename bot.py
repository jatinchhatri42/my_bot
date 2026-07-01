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
FIREBASE_CRED_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase_credentials.json")
FIREBASE_CRED_JSON  = os.getenv("FIREBASE_CREDENTIALS_JSON", "")  # full key file content, for hosted envs (Railway etc.)
LLM_TIMEOUT    = int(os.getenv("LLM_TIMEOUT", "15"))  # Groq is fast, no need for a 300s budget
OWNER_WELCOME  = os.getenv("OWNER_WELCOME_MESSAGE", "")  # optional custom intro line from you, the shop owner
RAZORPAY_KEY_ID     = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")
OWNER_CHAT_ID  = os.getenv("OWNER_CHAT_ID", "")  # your Telegram chat id, to get notified of new orders


def load_firebase_credentials():
    """
    Hosted platforms like Railway only have what's in your git repo — a local
    firebase_credentials.json you never committed (correctly, since it's a
    secret) simply won't exist there. So: prefer FIREBASE_CREDENTIALS_JSON
    (the full key file content pasted into an env var) when present, and
    fall back to a file on disk for local development. Fails with one clear
    message instead of crash-looping on the same traceback forever.
    """
    if FIREBASE_CRED_JSON:
        try:
            return credentials.Certificate(json.loads(FIREBASE_CRED_JSON))
        except json.JSONDecodeError as e:
            raise SystemExit(
                f"FIREBASE_CREDENTIALS_JSON is set but isn't valid JSON ({e}). "
                "Paste the ENTIRE contents of your serviceAccountKey.json file as the value."
            )
    if os.path.exists(FIREBASE_CRED_PATH):
        return credentials.Certificate(FIREBASE_CRED_PATH)
    raise SystemExit(
        f"No Firebase credentials found. Either set FIREBASE_CREDENTIALS_JSON "
        f"(paste the full service account JSON as one env var — required on "
        f"Railway/hosted platforms), or place the key file at "
        f"'{FIREBASE_CRED_PATH}' for local development."
    )


cred = load_firebase_credentials()
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
CATEGORY_BLACKLIST = {"default", "uncategorized", "none", ""}

def get_categories(limit: int = 8) -> list:
    """
    Distinct product categories for the /menu buttons. Firestore has no
    native DISTINCT, so we scan and dedupe in Python, with a 5-minute cache
    so opening /menu repeatedly doesn't re-scan the whole collection.
    Placeholder values like "Default" are skipped — they're a sign of
    products with no real category set, not an actual section of the shop.
    """
    import time
    now = time.time()
    if _category_cache["categories"] and now - _category_cache["ts"] < 300:
        return _category_cache["categories"]
    seen, ordered = set(), []
    docs = db.collection("products").limit(300).stream()
    for doc in docs:
        cat = (doc.to_dict() or {}).get("category", "").strip()
        if cat and cat.lower() not in CATEGORY_BLACKLIST and cat not in seen:
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
    """Returns 'faq', 'product', or 'chitchat' — pure keyword routing, no model call.
    Used as the safety-net fallback if the smart classifier below is unavailable
    or fails — the bot never goes fully silent just because Groq is down."""
    msg = user_msg.lower()
    if any(re.search(rf"\b{re.escape(k)}\b", msg) for k in FAQ_KEYWORDS):
        return "faq"
    if any(re.search(rf"\b{re.escape(k)}\b", msg) for k in PRODUCT_KEYWORDS):
        return "product"
    return "chitchat"


INTENT_SYSTEM_PROMPT = """You are an intent router for a bookstore Telegram bot.
Classify the customer's message into EXACTLY one of these labels:

- "buy"      — they want to purchase/order something right now, in ANY
               phrasing or spelling: "i wanna buy", "i want to buy", "buy",
               "i need it", "i'll take it", "lemme get this", "purchase",
               typos like "i wana by it" — all count as "buy".
- "product"  — browsing, searching, or asking about books: prices, ratings,
               categories, recommendations, "show me", "do you have...".
- "faq"      — shipping, returns, payment methods, delivery time, contact,
               warranty, policy questions.
- "chitchat" — greetings, small talk, anything that isn't the above.

Also extract a short "query": if the message names a specific book/title/
author/keyword, put it here (cleaned up, no filler words). If it doesn't
name anything specific (e.g. just "buy" or "i wanna buy"), leave query "".

Respond with ONLY a JSON object, nothing else, no markdown fences:
{"intent": "buy|product|faq|chitchat", "query": "<string>"}
"""


def classify_intent_smart(user_msg: str) -> dict:
    """
    LLM-based classification. This is the ONLY thing the LLM decides here —
    a label from a fixed set of 4, plus a search phrase echoed back from the
    user's own words. It can't introduce new facts or skip the deterministic
    Firebase fetch that happens afterward; it only picks which fetch to run.
    Falls back to the keyword classifier on any failure, so a Groq outage
    degrades the bot to "less smart routing", never "no routing".
    """
    fallback = {"intent": classify_intent(user_msg), "query": ""}
    if not GROQ_API_KEY:
        return fallback
    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0,
                "max_tokens": 60,
            },
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        intent = parsed.get("intent")
        if intent not in ("buy", "product", "faq", "chitchat"):
            return fallback
        return {"intent": intent, "query": (parsed.get("query") or "").strip()}
    except Exception:
        log.exception("Smart intent classification failed — using keyword fallback")
        return fallback


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
    return dedupe_products(results)


def dedupe_products(results: list) -> list:
    """
    Firestore can easily end up with two docs for the same book (re-imports,
    manual re-entry, etc). Same title + same price is treated as a duplicate
    listing and collapsed to one, so customers don't see "The Black Maria"
    twice in a row.
    """
    seen, deduped = set(), []
    for r in results:
        title = (r.get("title") or r.get("name") or "").strip().lower()
        price = str(r.get("price", ""))
        key = (title, price)
        if title and key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def fetch_faqs(user_msg: str) -> list:
    return search_faqs(user_msg)


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Render results
#   3a. Deterministic Python rendering (always correct, zero hallucination risk)
#   3b. Optional: Groq LLM "polish" pass — sees ONLY the fetched JSON, cannot add
#       facts that aren't in it. Used purely for friendlier natural-language
#       phrasing; if it fails or times out we silently keep the Python render.
# ══════════════════════════════════════════════════════════════════════════════

def build_product_cards(results: list, limit: int = 5) -> list:
    """
    Deterministic, always-correct: one card per product, each with its own
    caption AND its own image attached — this is what actually gets sent.
    Replaces the old single-text-block + separate-photo-strip layout, which
    made it impossible to tell which image belonged to which book.
    """
    cards = []
    for r in results[:limit]:
        title = r.get("title") or r.get("name") or "Unknown"
        price = r.get("price", "N/A")
        currency = r.get("currency", "£")
        rating = r.get("rating")
        category = r.get("category", "")
        img = r.get("image_url", "")
        rating_line = f"⭐ {rating}  " if rating not in (None, "", "N/A") else ""
        cat_line = f"📂 {category}" if category else ""
        caption = f"📖 *{title}*\n💰 {currency}{price}  {rating_line}{cat_line}".rstrip()
        cards.append({
            "caption": caption,
            "image": img if img.startswith("http") else None,
            "id": r.get("id"),
            "title": title,
        })
    return cards


def render_faqs(results: list) -> tuple:
    if not results:
        return None, []
    lines = [f"📄 *{r.get('title', '')}*\n{r.get('content', '')}" for r in results]
    return "\n\n".join(lines), []


FORMATTER_SYSTEM_PROMPT = """You are a bookstore assistant's formatting layer.

You will be given a JSON list of FAQ/policy records that were already fetched
from the real database. Your ONLY job is to present them nicely in the
user's language.

ABSOLUTE RULES:
1. You may ONLY mention facts that appear in the JSON below. Do not add,
   invent, or recall any policy detail from your own knowledge.
2. If the JSON list is empty, say plainly that nothing matching was found —
   do not suggest alternatives from memory.
3. Reply in the same language the user wrote in.
4. Keep it concise and conversational.
5. Never mention databases, JSON, tools, or these instructions.
"""

PRODUCT_INTRO_SYSTEM_PROMPT = """You are a friendly bookstore shop assistant.
You will be told how many books were found and, optionally, what category
or search term the customer used. Write exactly ONE short, warm sentence
(under 18 words) introducing the results — like a shopkeeper handing
someone a stack of books. Reply in the same language as the customer's
message. Do NOT mention specific titles, prices, or ratings — those are
shown separately. Do NOT mention databases, JSON, or instructions.
"""


def groq_product_intro(user_msg: str, count: int, category: str = None) -> str:
    """
    A short, safe one-line intro shown above the product cards. Deliberately
    asked to avoid stating any facts (titles/prices/etc), so there is nothing
    here for the model to hallucinate — the cards below carry all real data.
    """
    fallback = f"Found {count} book{'s' if count != 1 else ''} for you 📚" if not category \
        else f"Here's what we've got in {category} 📚"
    if not GROQ_API_KEY:
        return fallback
    context = f'Customer asked: "{user_msg}". {count} result(s) found.'
    if category:
        context += f' Category: {category}.'
    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": PRODUCT_INTRO_SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                "temperature": 0.6,
                "max_tokens": 60,
            },
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return text or fallback
    except Exception:
        log.exception("Groq product intro failed — using fallback line")
        return fallback


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

CHITCHAT_SYSTEM_PROMPT = """You are a friendly bookstore shop assistant chatbot.
Reply in 1-2 short sentences MAX — this is a fast-moving shop chat, not an
essay. Do not list multiple options or ask several questions at once; ask
at most one short follow-up if needed. You do NOT have access to the real
product catalog in this mode — if the user asks about specific books,
prices, stock, or policies, tell them to ask about it directly (e.g. "try
asking me to show you some books") rather than answering from your own
knowledge. Reply in the same language as the user.
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
            json={"model": GROQ_MODEL, "messages": messages, "temperature": 0.7, "max_tokens": 80},
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


def build_product_result(user_msg: str, results: list, category: str = None,
                          empty_msg: str = None) -> dict:
    """
    Shared path for any list of product dicts — used by both free-text
    search and the /menu inline-button flow, so menu taps and typed queries
    look identical. Returns {"intro": str, "cards": list} where cards is
    empty if nothing was found (caller sends just the intro/empty message).
    """
    if not results:
        return {"intro": empty_msg or "I couldn't find any matching products. "
                "Try a different keyword or category.", "cards": []}
    results = dedupe_products(results)
    cards = build_product_cards(results)
    intro = groq_product_intro(user_msg, len(cards), category=category)
    return {"intro": intro, "cards": cards}


def build_answer(user_msg: str, history: list, intent: str) -> tuple:
    """
    Returns (answer_text, image_urls) for FAQ/chitchat intents. Product and
    buy intents are handled separately in the Telegram layer below, since
    they render as cards/buttons, not prose — this function only covers
    the prose-style answers. `intent` is passed in (already decided by
    classify_intent_smart upstream) so we never classify the same message twice.
    """
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


def resolve_buy_target(query: str, last_results: list) -> list:
    """
    Figures out which product(s) a 'buy' intent refers to:
      1. If the message named something specific, search for it.
      2. Otherwise fall back to whatever was last shown in this chat
         (e.g. they viewed one book's price, then just said "buy").
    Returns a list of candidate products — empty if we genuinely can't tell.
    """
    if query:
        found = search_products(query, limit=5)
        if found:
            return found
    return last_results or []


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Checkout (Razorpay Payment Links) + order record in Firestore
# ══════════════════════════════════════════════════════════════════════════════

def create_order_record(product: dict, chat_id, telegram_user) -> str:
    """Always log the order attempt in Firestore first — this is the source
    of truth for what was ordered, independent of whether Razorpay or the
    payment step succeeds."""
    order = {
        "product_id": product.get("id"),
        "title": product.get("title") or product.get("name"),
        "price": product.get("price"),
        "currency": product.get("currency", ""),
        "chat_id": str(chat_id),
        "telegram_username": getattr(telegram_user, "username", None),
        "telegram_name": getattr(telegram_user, "full_name", None),
        "status": "pending_payment",
        "created_at": firestore.SERVER_TIMESTAMP,
    }
    ref = db.collection("orders").add(order)[1]
    log.info("Order created: %s for product %s", ref.id, product.get("id"))
    return ref.id


def razorpay_create_payment_link(product: dict, order_id: str) -> str | None:
    """
    Creates a Razorpay-hosted payment page and returns its URL. Returns None
    on any failure so the caller can fall back gracefully (never block the
    customer entirely just because the payment API hiccuped).
    """
    if not (RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET):
        return None
    title = product.get("title") or product.get("name") or "Book"
    try:
        price = float(product.get("price", 0))
    except (TypeError, ValueError):
        price = 0
    amount_paise = int(round(price * 100))
    if amount_paise <= 0:
        return None
    try:
        resp = requests.post(
            "https://api.razorpay.com/v1/payment_links",
            auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET),
            json={
                "amount": amount_paise,
                "currency": "INR",
                "description": title[:255],
                "reference_id": order_id,
                "notify": {"sms": False, "email": False},
                "reminder_enable": False,
            },
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("short_url")
    except Exception:
        log.exception("Razorpay payment link creation failed")
        return None


async def notify_owner_of_order(ctx: ContextTypes.DEFAULT_TYPE, product: dict, order_id: str,
                                 telegram_user) -> None:
    if not OWNER_CHAT_ID:
        return
    title = product.get("title") or product.get("name") or "Unknown"
    price = product.get("price", "N/A")
    who = f"@{telegram_user.username}" if getattr(telegram_user, "username", None) else telegram_user.full_name
    try:
        await ctx.bot.send_message(
            chat_id=OWNER_CHAT_ID,
            text=f"🛎️ New order ({order_id[:6]})\n📖 {title}\n💰 {price}\n👤 {who}",
        )
    except Exception:
        log.exception("Could not notify owner of new order")


# ══════════════════════════════════════════════════════════════════════════════
# Telegram handlers
# ══════════════════════════════════════════════════════════════════════════════

histories: dict = {}
last_shown: dict = {}  # chat_id -> last list of product dicts shown, for resolving bare "buy"

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
    """Send logic for prose answers (FAQ/chitchat): text, then any photos."""
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


async def deliver_products(chat_id, ctx: ContextTypes.DEFAULT_TYPE, result: dict, reply_markup=None):
    """
    Send a product result as a shop-style sequence: one short intro line,
    then one message PER product with its image attached directly to that
    product's own details, plus a "Buy Now" button so there's always a
    single unambiguous next action. Any extra keyboard (e.g. the /menu
    categories) goes in its own trailing message so it never collides with
    the per-product buy buttons.
    """
    intro, cards = result["intro"], result["cards"]
    await ctx.bot.send_message(chat_id=chat_id, text=intro)

    if not cards:
        if reply_markup:
            await ctx.bot.send_message(chat_id=chat_id, text="Want to browse more?", reply_markup=reply_markup)
        return

    await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
    for card in cards:
        buy_markup = None
        if card.get("id"):
            buy_markup = InlineKeyboardMarkup(
                [[InlineKeyboardButton("🛒 Buy Now", callback_data=f"buy:{card['id']}")]]
            )
        try:
            if card["image"]:
                await ctx.bot.send_photo(
                    chat_id=chat_id, photo=card["image"], caption=card["caption"],
                    parse_mode="Markdown", reply_markup=buy_markup,
                )
            else:
                await ctx.bot.send_message(
                    chat_id=chat_id, text=card["caption"],
                    parse_mode="Markdown", reply_markup=buy_markup,
                )
        except Exception as e:
            log.warning("Could not send product card: %s", e)
            await ctx.bot.send_message(chat_id=chat_id, text=card["caption"],
                                        parse_mode="Markdown", reply_markup=buy_markup)

    if reply_markup:
        await ctx.bot.send_message(chat_id=chat_id, text="Want to browse more?", reply_markup=reply_markup)


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
    keyboard = await asyncio.to_thread(build_menu_keyboard)

    if data == "show_all":
        results = await asyncio.to_thread(list_collection, "products", 8)
        last_shown[chat_id] = results
        result = await asyncio.to_thread(
            build_product_result, "show all products", results, None,
            "We don't have any products listed yet — check back soon!",
        )
        await deliver_products(chat_id, ctx, result, reply_markup=keyboard)

    elif data == "faq":
        results = await asyncio.to_thread(fetch_faqs, "shipping returns payment policy")
        text, _ = render_faqs(results)
        answer = text or "Ask me about shipping, returns, payment, or contact info anytime!"
        await deliver_answer(chat_id, ctx, answer, [], reply_markup=keyboard)

    elif data.startswith("cat:"):
        category = data.split(":", 1)[1]
        results = await asyncio.to_thread(search_by_field, "products", "category", category, 8)
        last_shown[chat_id] = results
        result = await asyncio.to_thread(
            build_product_result, f"products in category {category}", results, category,
            f"No products found in *{category}* right now.",
        )
        await deliver_products(chat_id, ctx, result, reply_markup=keyboard)

    else:
        await deliver_answer(chat_id, ctx, "Not sure what you picked — try /menu again.", [],
                              reply_markup=keyboard)


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id  = update.effective_chat.id
    user_msg = (update.message.text or "").strip()
    if not user_msg:
        return

    await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    history = get_history(chat_id)
    classification = await asyncio.to_thread(classify_intent_smart, user_msg)
    intent, query = classification["intent"], classification["query"]
    log.info("intent='%s' query='%s' for message: %r", intent, query, user_msg)

    try:
        if intent == "product":
            results = await asyncio.to_thread(fetch_products, query or user_msg)
            last_shown[chat_id] = results
            result = await asyncio.to_thread(
                build_product_result, user_msg, results, None,
                "I couldn't find any matching books in our catalog. Try a different keyword or category.",
            )
            update_history(chat_id, "user", user_msg)
            update_history(chat_id, "assistant", result["intro"])
            await deliver_products(chat_id, ctx, result)
            return

        if intent == "buy":
            candidates = await asyncio.to_thread(resolve_buy_target, query, last_shown.get(chat_id))
            update_history(chat_id, "user", user_msg)
            if not candidates:
                msg = "Which book would you like to buy? Tell me the title, or browse with /menu 🛍️"
                update_history(chat_id, "assistant", msg)
                await deliver_answer(chat_id, ctx, msg, [])
                return
            last_shown[chat_id] = candidates
            cards = await asyncio.to_thread(build_product_cards, candidates)
            intro = ("Tap *Buy Now* on the one you want 👇" if len(cards) > 1
                      else "Great choice — tap *Buy Now* to continue 👇")
            update_history(chat_id, "assistant", intro)
            await deliver_products(chat_id, ctx, {"intro": intro, "cards": cards})
            return

        answer, image_urls = await asyncio.to_thread(build_answer, user_msg, history, intent)
    except Exception:
        log.exception("Answer pipeline failed")
        answer, image_urls = "Sorry, something went wrong. Please try again.", []

    update_history(chat_id, "user", user_msg)
    update_history(chat_id, "assistant", answer)

    await deliver_answer(chat_id, ctx, answer, image_urls)


async def on_buy_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    Handles a 'Buy Now' tap: logs the order in Firestore, creates a Razorpay
    payment link for the exact item/price, and sends it as a Pay Now button.
    Falls back to a manual-order message if Razorpay isn't configured or the
    API call fails — the order is still recorded either way.
    """
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    doc_id = query.data.split(":", 1)[1]

    product = await asyncio.to_thread(get_document, "products", doc_id)
    if product.get("error"):
        await ctx.bot.send_message(chat_id=chat_id, text="Sorry, that item isn't available anymore.")
        return

    title = product.get("title") or product.get("name") or "this item"
    price = product.get("price", "N/A")
    currency = product.get("currency", "£")

    order_id = await asyncio.to_thread(create_order_record, product, chat_id, query.from_user)
    pay_url = await asyncio.to_thread(razorpay_create_payment_link, product, order_id)
    await notify_owner_of_order(ctx, product, order_id, query.from_user)

    if pay_url:
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("💳 Pay Now", url=pay_url)]])
        await ctx.bot.send_message(
            chat_id=chat_id,
            text=f"🛒 *{title}* — {currency}{price}\n\nTap below to complete payment securely:",
            parse_mode="Markdown",
            reply_markup=keyboard,
        )
    else:
        await ctx.bot.send_message(
            chat_id=chat_id,
            text=(f"🛒 *{title}* — {currency}{price}\n\n"
                  f"Your order ({order_id[:6]}) is logged! Online payment isn't set up yet — "
                  "we'll reach out shortly to arrange payment and delivery."),
            parse_mode="Markdown",
        )


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
    app.add_handler(CallbackQueryHandler(on_buy_button, pattern=r"^buy:"))
    app.add_handler(CallbackQueryHandler(on_menu_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    log.info("Bot started — model=%s (formatting-only, Groq) timeout=%ds", GROQ_MODEL, LLM_TIMEOUT)
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
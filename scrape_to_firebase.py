"""
scrape_to_firebase.py
─────────────────────
Scrapes books.toscrape.com (all 50 pages, ~1000 books) and
pushes each book into your Firestore `products` collection.

Also seeds a `documents` (FAQs) and `users` collection with
sample data so your Telegram bot has something to answer.

Run once:
    python scrape_to_firebase.py

Options:
    --pages N      only scrape N pages (default: all 50)
    --dry-run      print data without writing to Firebase
"""

import argparse
import os
import sys
import time

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--pages",   type=int, default=50, help="Number of pages to scrape (1-50)")
parser.add_argument("--dry-run", action="store_true",  help="Print only, no Firebase writes")
args = parser.parse_args()

# ── Firebase (skip import in dry-run) ─────────────────────────────────────────
db = None
if not args.dry_run:
    import firebase_admin
    from firebase_admin import credentials, firestore

    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase_credentials.json")
    if not os.path.exists(cred_path):
        print(f"❌  Firebase credentials not found at: {cred_path}")
        print("    Set FIREBASE_CREDENTIALS_PATH in .env or pass the right path.")
        sys.exit(1)

    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()

BASE       = "https://books.toscrape.com"
CATALOGUE  = f"{BASE}/catalogue"

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


# ══════════════════════════════════════════════════════════════════════════════
# Scraper
# ══════════════════════════════════════════════════════════════════════════════

def scrape_book_detail(detail_url: str) -> dict:
    """Fetch the book's detail page for description & category."""
    try:
        r = requests.get(detail_url, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        desc_tag = soup.select_one("#product_description ~ p")
        description = desc_tag.text.strip() if desc_tag else ""

        breadcrumbs = soup.select("ul.breadcrumb li")
        category = breadcrumbs[2].text.strip() if len(breadcrumbs) >= 3 else "General"

        return {"description": description, "category": category}
    except Exception:
        return {"description": "", "category": "General"}


def scrape_page(page_num: int) -> list[dict]:
    url = f"{CATALOGUE}/page-{page_num}.html"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  ⚠️  Page {page_num} failed: {e}")
        return []

    soup  = BeautifulSoup(r.text, "html.parser")
    cards = soup.select("article.product_pod")
    books = []

    for card in cards:
        # Title
        title = card.h3.a["title"]

        # Price  (strip £ and convert to float)
        price_raw = card.select_one(".price_color").text.strip()
        price = float(price_raw.replace("£", "").replace("Â", "").strip())

        # Star rating
        rating_word = card.p["class"][1]          # e.g. "Three"
        rating      = RATING_MAP.get(rating_word, 0)

        # Availability
        avail_text  = card.select_one(".availability").text.strip().lower()
        in_stock    = "in stock" in avail_text

        # Image URL  (fix relative path)
        img_src = card.img["src"].replace("../../", "")
        image_url = f"{BASE}/{img_src}"

        # Detail page URL
        href       = card.h3.a["href"].replace("../../", "")
        detail_url = f"{CATALOGUE}/{href}"

        books.append({
            "title":      title,
            "price":      price,
            "currency":   "GBP",
            "rating":     rating,
            "in_stock":   in_stock,
            "image_url":  image_url,
            "detail_url": detail_url,
        })

    return books


# ══════════════════════════════════════════════════════════════════════════════
# FAQ / Document seeds
# ══════════════════════════════════════════════════════════════════════════════

FAQ_DOCS = [
    {
        "title": "Shipping Policy",
        "content": (
            "We offer free standard shipping on all orders above £20. "
            "Standard delivery takes 3–5 working days. "
            "Express delivery (1–2 days) is available for £4.99. "
            "International shipping is available to 30+ countries."
        ),
        "type": "policy",
        "tags": ["shipping", "delivery", "order"],
    },
    {
        "title": "Return & Refund Policy",
        "content": (
            "Books can be returned within 30 days of purchase in their original condition. "
            "To initiate a return, contact support@bookstore.com with your order ID. "
            "Refunds are processed within 5–7 business days."
        ),
        "type": "policy",
        "tags": ["return", "refund", "policy"],
    },
    {
        "title": "How to Track Your Order",
        "content": (
            "Once your order is dispatched, you will receive a tracking number by email. "
            "You can track your order at our website under 'My Orders'. "
            "For help, contact support@bookstore.com."
        ),
        "type": "faq",
        "tags": ["tracking", "order", "shipment"],
    },
    {
        "title": "Payment Methods",
        "content": (
            "We accept Visa, Mastercard, PayPal, UPI, and net banking. "
            "All transactions are secured with 256-bit SSL encryption. "
            "Cash on delivery is available in select areas."
        ),
        "type": "faq",
        "tags": ["payment", "card", "upi"],
    },
    {
        "title": "Contact Us",
        "content": (
            "Customer support is available Monday–Saturday, 9 AM to 6 PM IST. "
            "Email: support@bookstore.com | Phone: +91-98765-43210. "
            "Average response time: under 2 hours."
        ),
        "type": "contact",
        "tags": ["contact", "support", "help"],
    },
]

SAMPLE_USERS = [
    {"name": "Aarav Shah",    "email": "aarav@example.com",   "city": "Mumbai",    "member_since": "2022-03-15", "orders": 12},
    {"name": "Priya Menon",   "email": "priya@example.com",   "city": "Bangalore", "member_since": "2021-11-02", "orders": 7},
    {"name": "Rohit Sharma",  "email": "rohit@example.com",   "city": "Delhi",     "member_since": "2023-01-20", "orders": 3},
    {"name": "Sneha Patil",   "email": "sneha@example.com",   "city": "Pune",      "member_since": "2020-06-08", "orders": 25},
    {"name": "Vikram Nair",   "email": "vikram@example.com",  "city": "Chennai",   "member_since": "2023-07-11", "orders": 1},
]


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    pages   = min(max(args.pages, 1), 50)
    dry_run = args.dry_run

    print(f"\n{'[DRY RUN] ' if dry_run else ''}📚  Scraping books.toscrape.com ({pages} page(s))…\n")

    all_books  = []
    total_written = 0

    for page in range(1, pages + 1):
        print(f"  Page {page}/{pages}…", end=" ", flush=True)
        books = scrape_page(page)

        for book in books:
            # Fetch detail page for description + category
            extra = scrape_book_detail(book["detail_url"])
            book.update(extra)

            if dry_run:
                print(f"\n    📖 {book['title'][:50]} | £{book['price']} | ⭐{book['rating']} | {book['category']}")
                print(f"       🖼  {book['image_url']}")
            else:
                db.collection("products").add(book)
                total_written += 1

            all_books.append(book)
            time.sleep(0.05)   # polite crawl delay

        print(f"✓ {len(books)} books" if not dry_run else "")
        time.sleep(0.3)

    print(f"\n{'[DRY RUN] ' if dry_run else ''}✅  Books done: {len(all_books)} total")

    # ── FAQs / documents ──
    print(f"\n📄  {'Printing' if dry_run else 'Uploading'} FAQ documents…")
    for doc in FAQ_DOCS:
        if dry_run:
            print(f"   • {doc['title']}")
        else:
            db.collection("documents").add(doc)
    print(f"   ✓ {len(FAQ_DOCS)} documents")

    # ── Users ──
    print(f"\n👤  {'Printing' if dry_run else 'Uploading'} sample users…")
    for user in SAMPLE_USERS:
        if dry_run:
            print(f"   • {user['name']} ({user['city']})")
        else:
            db.collection("users").add(user)
    print(f"   ✓ {len(SAMPLE_USERS)} users")

    if not dry_run:
        print(f"\n🎉  All done! Firestore now has:")
        print(f"    • {total_written} products  (collection: products)")
        print(f"    • {len(FAQ_DOCS)} documents (collection: documents)")
        print(f"    • {len(SAMPLE_USERS)} users     (collection: users)")
        print(f"\nStart your Telegram bot:  python bot.py")
    else:
        print(f"\n[DRY RUN complete — no data was written to Firebase]")
        print("Remove --dry-run to actually upload.")


if __name__ == "__main__":
    main()

# 🤖 Telegram Bot — Ollama (qwen2.5:14b) + Firebase Firestore

Ask questions in Telegram → Ollama searches your Firestore → replies with text & images.

---

## 📁 Project Structure

```
telegram_bot/
├── bot.py               ← Main bot (run this)
├── seed_firebase.py     ← Populate Firestore with sample data (run once)
├── requirements.txt     ← Python dependencies
├── .env                 ← Your secrets (never commit this)
└── firebase_credentials.json  ← Firebase service account key (never commit)
```

---

## ⚙️ Step-by-Step Setup

### 1. Prerequisites

Make sure you have:
- Python 3.11+
- Ollama installed and running with qwen2.5:14b pulled

```bash
# Pull the model if you haven't already
ollama pull qwen2.5:14b

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

---

### 2. Get Your Telegram Bot Token

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` and follow the prompts
3. Copy the token it gives you (looks like `123456789:ABCdef...`)

---

### 3. Get Your Firebase Credentials

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Open your project → **Project Settings** → **Service Accounts**
3. Click **"Generate new private key"** → Download the JSON file
4. Rename it to `firebase_credentials.json` and place it in this folder

---

### 4. Configure `.env`

Edit the `.env` file:

```env
TELEGRAM_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ   ← paste your token
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:14b
FIREBASE_CREDENTIALS_PATH=firebase_credentials.json
```

---

### 5. Install Dependencies

```bash
cd telegram_bot
pip install -r requirements.txt
```

---

### 6. (Optional) Seed Sample Data

If you want to test with sample products first:

```bash
python seed_firebase.py
```

---

### 7. Run the Bot

```bash
python bot.py
```

You should see:
```
Bot started — model: qwen2.5:14b  ollama: http://localhost:11434
```

Now open Telegram, find your bot, and send `/start`!

---

## 💬 Example Conversations

| You type | Bot does |
|----------|----------|
| "Show me headphones" | Searches `products` collection, returns name + price + image |
| "Do you have yoga mats in stock?" | Queries Firestore, checks `in_stock` field |
| "What is your return policy?" | Fetches from `documents` collection |
| "List all electronics" | Searches by category |

---

## 🗂️ Firestore Collections Expected

| Collection | Fields |
|------------|--------|
| `products` | `name`, `description`, `price`, `currency`, `category`, `tags[]`, `image_url`, `in_stock` |
| `documents` | `title`, `content`, `type` |

You can add any other collections — just ask the bot naturally and Ollama will use the `search_collection` or `list_collection` tools to explore them.

---

## 🛠️ Customisation

**Add a new Firestore collection?**
No code changes needed — the bot has generic tools (`list_collection`, `search_collection`, `get_document`) that work on any collection.

**Change the AI model?**
Edit `OLLAMA_MODEL` in `.env`.

**Deploy to a server?**
- Copy the folder to your VPS
- Change `OLLAMA_URL` to point to your Ollama instance
- Run with `nohup python bot.py &` or use `systemd`/`screen`

---

## 🔒 Security

- **Never commit** `.env` or `firebase_credentials.json` to Git
- Add both to `.gitignore`:
  ```
  .env
  firebase_credentials.json
  ```

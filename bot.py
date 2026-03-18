import os
import logging
import asyncio
import requests
import feedparser
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from openai import AsyncOpenAI

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
RSS_URL = os.getenv("RSS_URL")

# Validate required environment variables
if not all([TELEGRAM_TOKEN, OPENROUTER_API_KEY, RSS_URL]):
    logger.error("Missing required environment variables. Please check your .env file.")
    exit(1)

# Initialize the Async OpenRouter Client
llm_client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the command /start is issued."""
    welcome_message = (
        "👋 Bonjour ! Je suis ton Sourceur de Stages personnel.\n\n"
        "Voici comment je fonctionne :\n"
        "1. Utilise la commande /setcv pour m'envoyer ton CV au format Markdown (.md).\n"
        "2. Utilise la commande /parse pour que je lise les dernières offres de stage "
        "et trouve les meilleures correspondances avec ton profil.\n\n"
        "Prêt ? Envoie /setcv pour commencer !"
    )
    await update.message.reply_text(welcome_message)


async def setcv_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Prompt the user to send their CV document."""
    await update.message.reply_text(
        "Veuillez m'envoyer votre CV au format Markdown (fichier avec l'extension .md)."
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process the uploaded Markdown document and store its content."""
    document = update.message.document

    if not document.file_name.endswith(".md"):
        await update.message.reply_text("❌ Le fichier doit être au format Markdown (.md).")
        return

    try:
        # Download the file via Telegram API
        file = await context.bot.get_file(document.file_id)
        file_byte_array = await file.download_as_bytearray()

        # Read content as UTF-8 string
        cv_content = file_byte_array.decode("utf-8")

        # Store the CV content securely in the user's session data
        context.user_data["cv_content"] = cv_content

        await update.message.reply_text(
            "✅ Votre CV a été enregistré avec succès !\n"
            "Vous pouvez maintenant utiliser la commande /parse pour chercher des stages."
        )
    except Exception as e:
        logger.error(f"Error handling document upload: {e}")
        await update.message.reply_text("❌ Une erreur est survenue lors de la lecture de votre fichier.")


def fetch_rss_feed(url: str) -> list:
    """Fetch and parse the RSS feed synchronously."""
    try:
        # Using requests as standard HTTP client to fetch the raw XML
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parsing the raw content with feedparser
        feed = feedparser.parse(response.content)
        return feed.entries[:5]
    except Exception as e:
        logger.error(f"Error fetching RSS feed: {e}")
        return []


async def parse_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fetch RSS, evaluate offers with the LLM against the user's CV, and send results."""
    cv_content = context.user_data.get("cv_content")

    if not cv_content:
        await update.message.reply_text(
            "⚠️ Je n'ai pas de CV en mémoire. Veuillez utiliser /setcv pour m'envoyer votre CV d'abord."
        )
        return

    await update.message.reply_text("🔍 Recherche des dernières offres et analyse en cours. Veuillez patienter...")

    # Run the synchronous HTTP request in a separate thread to prevent blocking the event loop
    entries = await asyncio.to_thread(fetch_rss_feed, RSS_URL)

    if not entries:
        await update.message.reply_text("❌ Impossible de récupérer les offres pour le moment. Le flux est peut-être vide ou inaccessible.")
        return

    # Construct the textual representation of the offers
    offers_text = "\n\n".join(
        f"Offre {idx}:\nTitre: {entry.get('title', 'N/A')}\nLien: {entry.get('link', 'N/A')}\nRésumé: {entry.get('summary', 'N/A')}"
        for idx, entry in enumerate(entries, start=1)
    )

    # Prepare the prompt for the LLM
    prompt = f"""
You are an expert technical recruiter matching candidates to internships.
Here is the candidate's CV:
---
{cv_content}
---
Here are the 5 latest internship offers:
---
{offers_text}
---
Task:
1. Select the 2 to 3 best internship offers that strictly match the candidate's CV.
2. Reply ENTIRELY IN FRENCH.
3. For each selected offer, include the Title and the Link.
4. Provide a justification for your choice detailing what matches (skills, tech stack) and what is missing based on the CV.
"""

    try:
        # Request completion from OpenRouter
        response = await llm_client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "You are a helpful internship sourcing assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        llm_response = response.choices[0].message.content
        await update.message.reply_text(llm_response)

    except Exception as e:
        logger.error(f"Error calling LLM via OpenRouter: {e}")
        await update.message.reply_text("❌ Une erreur est survenue lors de l'analyse des offres avec l'IA.")


def main() -> None:
    """Initialize and run the Telegram bot."""
    # Build the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Register command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("setcv", setcv_command))
    application.add_handler(CommandHandler("parse", parse_command))

    # Register message handler for Markdown documents
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Start the Bot
    logger.info("Bot is starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
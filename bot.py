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
        "👋 Bonjour ! Je suis ton Assistant Carrière multi-fonctions.\n\n"
        "Voici ce que je peux faire :\n"
        "1. Documents : /setcv, /setjob, /setinterviewer pour enregistrer tes informations.\n"
        "2. Sourcing : /parse pour analyser les dernières offres de stage avec ton CV.\n"
        "3. Coaching : /coach pour analyser tes lacunes, /linkedin pour un message d'approche.\n"
        "4. Entretien : /interview pour t'entraîner avec un recruteur virtuel, puis /stop_interview pour terminer.\n\n"
        "Commence par envoyer /setcv !"
    )
    await update.message.reply_text(welcome_message)


async def set_cv_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set expected document to CV."""
    context.user_data["expected_doc"] = "cv_content"
    await update.message.reply_text("Veuillez m'envoyer votre CV au format Markdown (.md).")


async def set_job_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set expected document to Job Description."""
    context.user_data["expected_doc"] = "job_content"
    await update.message.reply_text("Veuillez m'envoyer la description du poste visé au format Markdown (.md).")


async def set_interviewer_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set expected document to Interviewer Profile."""
    context.user_data["expected_doc"] = "interviewer_content"
    await update.message.reply_text("Veuillez m'envoyer le profil du recruteur au format Markdown (.md).")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process the uploaded Markdown document and store it in the expected state."""
    expected_doc = context.user_data.get("expected_doc")
    
    if not expected_doc:
        await update.message.reply_text("Je n'attends aucun document. Utilisez d'abord /setcv, /setjob ou /setinterviewer.")
        return

    document = update.message.document
    if not document.file_name.endswith(".md"):
        await update.message.reply_text("❌ Erreur : Le fichier doit être au format Markdown (.md).")
        return

    try:
        file = await context.bot.get_file(document.file_id)
        file_byte_array = await file.download_as_bytearray()
        content = file_byte_array.decode("utf-8")
        
        context.user_data[expected_doc] = content
        context.user_data["expected_doc"] = None
        
        await update.message.reply_text("✅ Document reçu et enregistré avec succès dans la mémoire de session !")
    except Exception as e:
        logger.error(f"Error handling document download/decode: {e}")
        await update.message.reply_text("Une erreur est survenue lors de la lecture de votre fichier.")


def fetch_rss_feed(url: str) -> list:
    """Fetch and parse the RSS feed with anti-403 headers."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        return feed.entries[:5]
    except Exception as e:
        logger.error(f"Error fetching RSS feed: {e}")
        return []


async def parse_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fetch RSS, match with CV using LLM."""
    cv_content = context.user_data.get("cv_content")
    
    if not cv_content:
        await update.message.reply_text("⚠️ Veuillez d'abord envoyer votre CV avec /setcv.")
        return

    await update.message.reply_text("🔍 Recherche des offres et analyse en cours...")
    
    entries = await asyncio.to_thread(fetch_rss_feed, RSS_URL)
    if not entries:
        await update.message.reply_text("❌ Impossible de récupérer les offres pour le moment.")
        return

    offers_text = "\n\n".join(
        f"Titre: {entry.get('title', 'N/A')}\nLien: {entry.get('link', 'N/A')}\nRésumé: {entry.get('summary', 'N/A')}"
        for entry in entries
    )

    prompt = (
        f"Voici mon CV :\n{cv_content}\n\n"
        f"Voici les offres :\n{offers_text}\n\n"
        "Sélectionne les 2 à 3 meilleures offres qui matchent avec mon CV. "
        "Justifie ton choix (ce qui matche, ce qui manque). Réponds en français de manière concise."
    )

    try:
        response = await llm_client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        # Coupure de sécurité à 4000 caractères pour Telegram
        safe_response = response.choices[0].message.content[:4000]
        await update.message.reply_text(safe_response)
    except Exception as e:
        logger.error(f"Error calling LLM for /parse: {e}")
        await update.message.reply_text(f"❌ Erreur de l'IA lors du sourcing : {str(e)}")


async def coach_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Analyze CV against Job description."""
    cv_content = context.user_data.get("cv_content")
    job_content = context.user_data.get("job_content")
    
    if not cv_content or not job_content:
        await update.message.reply_text("⚠️ Il me faut votre CV (/setcv) ET la description du poste (/setjob) pour le coaching.")
        return

    await update.message.reply_text("🧠 Analyse croisée de votre profil par rapport au poste en cours...")
    
    prompt = (
        f"CV :\n{cv_content}\n\n"
        f"Poste visé :\n{job_content}\n\n"
        "Fais une analyse critique des lacunes du CV par rapport à ce poste et propose des améliorations concrètes. Réponds en français et sois synthétique (maximum 400 mots)."
    )

    try:
        response = await llm_client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        safe_response = response.choices[0].message.content[:4000]
        await update.message.reply_text(safe_response)
    except Exception as e:
        logger.error(f"Error calling LLM for /coach: {e}")
        await update.message.reply_text(f"❌ Erreur détaillée de l'IA (Coaching) : {str(e)}")


async def linkedin_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate a LinkedIn outreach message."""
    cv_content = context.user_data.get("cv_content")
    job_content = context.user_data.get("job_content")
    
    if not cv_content or not job_content:
        await update.message.reply_text("⚠️ Il me faut votre CV (/setcv) ET la description du poste (/setjob) pour rédiger le message.")
        return

    await update.message.reply_text("✍️ Rédaction du message d'approche en cours...")
    
    prompt = (
        f"CV :\n{cv_content}\n\n"
        f"Poste :\n{job_content}\n\n"
        "Rédige un message d'approche LinkedIn professionnel et accrocheur pour le recruteur de ce poste, en mettant en avant mon profil. Réponds en français de manière directe."
    )

    try:
        response = await llm_client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        safe_response = response.choices[0].message.content[:4000]
        await update.message.reply_text(safe_response)
    except Exception as e:
        logger.error(f"Error calling LLM for /linkedin: {e}")
        await update.message.reply_text(f"❌ Erreur détaillée de l'IA (LinkedIn) : {str(e)}")


async def interview_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start the interview simulator."""
    cv_content = context.user_data.get("cv_content")
    job_content = context.user_data.get("job_content")
    interviewer_content = context.user_data.get("interviewer_content")

    if not all([cv_content, job_content, interviewer_content]):
        await update.message.reply_text("⚠️ Pour l'entretien, il faut envoyer CV (/setcv), Poste (/setjob) et Profil du Recruteur (/setinterviewer).")
        return

    context.user_data["is_interviewing"] = True
    await update.message.reply_text("🎙️ Le simulateur démarre. Le recruteur prépare sa première question...")

    system_prompt = (
        f"Tu es le recruteur décrit ici : {interviewer_content}.\n"
        f"Poste à pourvoir : {job_content}.\n"
        f"CV du candidat : {cv_content}.\n\n"
        "Mène un entretien réaliste en français. "
        "À chaque tour, donne un très court feedback sur la réponse précédente (entre parenthèses), puis pose ta prochaine question. "
        "Commence maintenant par te présenter et poser ta première question."
    )

    context.user_data["chat_history"] = [{"role": "system", "content": system_prompt}]

    try:
        response = await llm_client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=context.user_data["chat_history"],
            temperature=0.7,
        )
        llm_reply = response.choices[0].message.content
        context.user_data["chat_history"].append({"role": "assistant", "content": llm_reply})
        
        await update.message.reply_text(llm_reply[:4000])
    except Exception as e:
        logger.error(f"Error calling LLM for /interview initialization: {e}")
        context.user_data["is_interviewing"] = False
        await update.message.reply_text(f"❌ Erreur lors du lancement de l'entretien : {str(e)}")


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle regular text messages during an interview."""
    if not context.user_data.get("is_interviewing"):
        await update.message.reply_text("Je ne comprends pas cette commande. Utilisez /start pour voir le menu.")
        return

    user_text = update.message.text
    context.user_data["chat_history"].append({"role": "user", "content": user_text})

    try:
        response = await llm_client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=context.user_data["chat_history"],
            temperature=0.7,
        )
        llm_reply = response.choices[0].message.content
        context.user_data["chat_history"].append({"role": "assistant", "content": llm_reply})
        
        await update.message.reply_text(llm_reply[:4000])
    except Exception as e:
        logger.error(f"Error calling LLM during interview: {e}")
        await update.message.reply_text(f"❌ Le recruteur a un problème technique : {str(e)}")


async def stop_interview_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Stop the interview and generate debriefing."""
    if not context.user_data.get("is_interviewing"):
        await update.message.reply_text("Aucun entretien en cours.")
        return

    await update.message.reply_text("🛑 Entretien terminé. Génération du débriefing en cours...")
    
    context.user_data["chat_history"].append({
        "role": "user",
        "content": "L'entretien est terminé. Fais-moi un débriefing global très structuré : points forts, points faibles, et une note sur 10."
    })

    try:
        response = await llm_client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=context.user_data["chat_history"],
            temperature=0.7,
        )
        safe_response = response.choices[0].message.content[:4000]
        await update.message.reply_text(safe_response)
    except Exception as e:
        logger.error(f"Error calling LLM for debriefing: {e}")
        await update.message.reply_text(f"❌ Erreur lors du débriefing : {str(e)}")
    finally:
        context.user_data["is_interviewing"] = False
        context.user_data["chat_history"] = []


def main() -> None:
    """Initialize and run the bot."""
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("setcv", set_cv_command))
    application.add_handler(CommandHandler("setjob", set_job_command))
    application.add_handler(CommandHandler("setinterviewer", set_interviewer_command))
    application.add_handler(CommandHandler("parse", parse_command))
    application.add_handler(CommandHandler("coach", coach_command))
    application.add_handler(CommandHandler("linkedin", linkedin_command))
    application.add_handler(CommandHandler("interview", interview_command))
    application.add_handler(CommandHandler("stop_interview", stop_interview_command))
    
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    logger.info("Bot is starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
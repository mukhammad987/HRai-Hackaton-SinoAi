"""
Telegram Bot Service for HR AI Assistant
"""

import os
import logging
from typing import Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
import httpx

from matching import get_matching_service
from database import get_session, Candidate, Vacancy, Match
from ai_hr_manager import get_ai_manager

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class HRBotService:
    """HR AI Assistant Telegram Bot Service"""
    
    def __init__(self, token: str):
        self.token = token
        self.matching_service = get_matching_service()
        self.application = None
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        welcome_message = f"""
👋 Привет, {user.first_name}!

Я - HR AI Ассистент. Помогу найти подходящую вакансию!

📋 **КАК ОТПРАВИТЬ РЕЗЮМЕ:**

1️⃣ **PDF файл** (рекомендуется) 📎
   • Просто прикрепите файл резюме
   • AI автоматически проанализирует его

2️⃣ **Текстовое сообщение** 💬
   • Напишите резюме прямо в чат
   • Минимум 100 символов

✅ **ЧТО ДОЛЖНО БЫТЬ В РЕЗЮМЕ:**

📌 **Обязательно укажите:**
• **Имя и контакты** - ФИО, телефон, email
• **Навыки** - технологии, языки программирования, инструменты
• **Опыт работы** - должности, компании, обязанности
• **Образование** - учебные заведения, специальность

📌 **Желательно добавить:**
• Проекты и достижения
• Сертификаты и курсы
• Уровень владения навыками
• Желаемая должность

💡 **ПРИМЕР РЕЗЮМЕ:**
"Иванов Иван Иванович
Python разработчик, 3 года опыта
Навыки: Python, Django, FastAPI, PostgreSQL, Docker, Git
Опыт: Backend разработчик в Tech Corp (2021-2024)
Email: ivan@example.com"

🎯 **ЧТО Я СДЕЛАЮ:**
✓ Проанализирую ваши навыки с помощью AI
✓ Найду подходящие вакансии
✓ Покажу процент соответствия (0-100%)
✓ Выделю совпадающие навыки

📝 **КОМАНДЫ:**
/vacancies - Все вакансии
/my_profile - Мой профиль
/help - Подробная помощь
/chat - 💬 Чат с AI HR менеджером (для работодателей)

📎 **Отправьте резюме прямо сейчас!**
"""
        await update.message.reply_text(welcome_message)
        
        db = get_session()
        try:
            candidate = db.query(Candidate).filter(
                Candidate.telegram_user_id == str(user.id)
            ).first()
            
            if not candidate:
                candidate = Candidate(
                    telegram_user_id=str(user.id),
                    name=f"{user.first_name} {user.last_name or ''}".strip()
                )
                db.add(candidate)
                db.commit()
        finally:
            db.close()
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
📖 **ПОДРОБНАЯ ИНСТРУКЦИЯ**

🔹 **ДЛЯ СОИСКАТЕЛЕЙ:**

**Шаг 1: Подготовьте резюме**
Включите обязательную информацию:
• Имя, контакты (телефон, email)
• Навыки и технологии
• Опыт работы (компании, должности, период)
• Образование

**Шаг 2: Отправьте резюме**
Два варианта:
📎 PDF файл - просто прикрепите файл
💬 Текст - напишите в чат (мин. 100 символов)

**Шаг 3: Получите результат**
AI покажет топ-5 вакансий с:
• Процентом соответствия (0-100%)
• Совпадающими навыками
• Деталями вакансии

**Шаг 4: Обновите резюме**
Для новых подборок просто отправьте обновленное резюме

🔹 **ДЛЯ РАБОТОДАТЕЛЕЙ:**

Используйте /chat для общения с AI HR менеджером:
• Помощь в составлении вакансий
• Консультации по требованиям
• Объяснение результатов подбора

💡 **СОВЕТЫ:**
✓ PDF формат читается лучше всего
✓ Чем подробнее резюме - тем точнее подбор
✓ Указывайте конкретные навыки и технологии
✓ Обновляйте резюме при смене опыта

📊 **ОЦЕНКА СООТВЕТСТВИЯ:**
🟢 70-100% - Отличное соответствие
🟡 50-69% - Хорошее соответствие
🟠 30-49% - Частичное соответствие
🔴 0-29% - Слабое соответствие

❓ Вопросы? Пишите /start для начала!
"""
        await update.message.reply_text(help_text)
    
    async def vacancies_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show all available vacancies"""
        db = get_session()
        try:
            vacancies = db.query(Vacancy).filter(Vacancy.is_active == True).all()
            
            if not vacancies:
                await update.message.reply_text("📭 К сожалению, сейчас нет доступных вакансий.")
                return
            
            message = "💼 Доступные вакансии:\n\n"
            for i, vac in enumerate(vacancies[:10], 1):
                message += f"{i}. {vac.title}\n"
                message += f"   🏢 {vac.company or 'Компания'}\n"
                message += f"   📍 {vac.location or 'Не указано'}\n"
                message += f"   ⏰ {vac.employment_type or 'Полная занятость'}\n\n"
            
            message += "\n📎 Отправьте резюме для подбора подходящих вакансий!"
            
            await update.message.reply_text(message)
        finally:
            db.close()
    
    async def my_profile_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show user profile"""
        user = update.effective_user
        db = get_session()
        try:
            candidate = db.query(Candidate).filter(
                Candidate.telegram_user_id == str(user.id)
            ).first()
            
            if not candidate or not candidate.resume_text:
                await update.message.reply_text(
                    "📝 У вас еще нет профиля.\n\n"
                    "Отправьте резюме, чтобы создать профиль!"
                )
                return
            
            profile_text = f"""
👤 Ваш профиль:

Имя: {candidate.name or 'Не указано'}
Контакт: {candidate.contact or 'Не указан'}

📄 Резюме получено: ✅
📅 Последнее обновление: {candidate.last_interaction_at.strftime('%d.%m.%Y')}

💡 Отправьте новое резюме для обновления профиля
"""
            await update.message.reply_text(profile_text)
        finally:
            db.close()
    
    async def chat_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start AI HR Manager chat for employers"""
        chat_intro = """
💬 Добро пожаловать в чат с AI HR Менеджером!

Я помогу вам:
• Создать эффективные вакансии
• Сформулировать требования к кандидатам
• Разобраться с результатами AI подбора
• Оптимизировать процесс найма

❓ Просто напишите ваш вопрос, и я отвечу!

Примеры вопросов:
"Как составить вакансию для Python разработчика?"
"Что означает процент соответствия 75%?"
"Какие навыки указать в вакансии ML инженера?"

Введите /stop чтобы выйти из чата.
"""
        await update.message.reply_text(chat_intro)
        
        context.user_data['ai_chat_active'] = True
        context.user_data['chat_history'] = []
    
    async def stop_chat_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Stop AI chat mode"""
        context.user_data['ai_chat_active'] = False
        context.user_data['chat_history'] = []
        await update.message.reply_text(
            "👋 Чат с AI HR менеджером завершен.\n\n"
            "Используйте /chat чтобы начать снова!"
        )
    
    async def handle_ai_chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle messages in AI chat mode"""
        user_message = update.message.text
        
        await update.message.reply_text("💭 Думаю...")
        
        try:
            ai_manager = get_ai_manager()
            chat_history = context.user_data.get('chat_history', [])
            
            response = ai_manager.chat(user_message, chat_history)
            
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": response})
            
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
            
            context.user_data['chat_history'] = chat_history
            
            await update.message.reply_text(response)
            
        except Exception as e:
            logger.error(f"AI chat error: {e}")
            await update.message.reply_text(
                "❌ Ошибка при общении с AI. Попробуйте еще раз."
            )
    
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle PDF document (resume)"""
        user = update.effective_user
        document = update.message.document
        
        if not document.file_name.lower().endswith('.pdf'):
            await update.message.reply_text(
                "⚠️ Пожалуйста, отправьте файл в формате PDF"
            )
            return
        
        await update.message.reply_text("⏳ Обрабатываю ваше резюме...")
        
        try:
            file = await context.bot.get_file(document.file_id)
            file_bytes = await file.download_as_bytearray()
            
            resume_text = self.matching_service.parse_pdf_bytes(bytes(file_bytes))
            
            await self.process_resume(update, user, resume_text)
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            await update.message.reply_text(
                "❌ Ошибка при обработке PDF. Попробуйте отправить текстовое резюме."
            )
    
    async def handle_text_or_chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text message - either AI chat or resume"""
        if context.user_data.get('ai_chat_active', False):
            await self.handle_ai_chat(update, context)
        else:
            await self.handle_text(update, context)
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text message (text resume)"""
        user = update.effective_user
        text = update.message.text
        
        if len(text) < 100:
            await update.message.reply_text(
                "⚠️ Резюме слишком короткое. Пожалуйста, укажите больше информации о себе и своих навыках."
            )
            return
        
        await update.message.reply_text("⏳ Анализирую ваше резюме...")
        
        await self.process_resume(update, user, text)
    
    async def process_resume(self, update: Update, user, resume_text: str):
        """Process resume and find matching vacancies"""
        db = get_session()
        try:
            candidate = db.query(Candidate).filter(
                Candidate.telegram_user_id == str(user.id)
            ).first()
            
            if not candidate:
                candidate = Candidate(
                    telegram_user_id=str(user.id),
                    name=f"{user.first_name} {user.last_name or ''}".strip()
                )
                db.add(candidate)
            
            candidate.resume_text = resume_text
            resume_embedding = self.matching_service.get_embedding(resume_text)
            candidate.set_embedding(resume_embedding)
            db.commit()
            
            vacancies = db.query(Vacancy).filter(Vacancy.is_active == True).all()
            
            if not vacancies:
                await update.message.reply_text(
                    "✅ Резюме получено!\n\n"
                    "📭 К сожалению, сейчас нет доступных вакансий.\n"
                    "Мы сообщим, когда появятся подходящие позиции!"
                )
                return
            
            vacancy_data = [
                (vac.id, vac.get_embedding(), vac.title)
                for vac in vacancies if vac.get_embedding() is not None
            ]
            
            if not vacancy_data:
                await update.message.reply_text(
                    "⚠️ Вакансии еще обрабатываются. Попробуйте позже."
                )
                return
            
            matches = self.matching_service.match_candidate_to_vacancies(
                resume_embedding, vacancy_data, top_k=5
            )
            
            for match_data in matches:
                match_record = Match(
                    candidate_id=candidate.id,
                    vacancy_id=match_data['vacancy_id'],
                    match_score=match_data['match_score']
                )
                db.add(match_record)
            
            db.commit()
            
            response = "✅ Резюме обработано!\n\n"
            response += "🎯 Подходящие вакансии:\n\n"
            
            for i, match in enumerate(matches, 1):
                vacancy = db.query(Vacancy).filter(Vacancy.id == match['vacancy_id']).first()
                score = match['match_score']
                
                if score >= 70:
                    emoji = "🟢"
                    rating = "Отличное соответствие"
                elif score >= 50:
                    emoji = "🟡"
                    rating = "Хорошее соответствие"
                else:
                    emoji = "🟠"
                    rating = "Частичное соответствие"
                
                response += f"{i}. {emoji} {vacancy.title}\n"
                response += f"   🏢 {vacancy.company or 'Компания'}\n"
                response += f"   📍 {vacancy.location or 'Не указано'}\n"
                response += f"   📊 Соответствие: {score:.1f}% - {rating}\n\n"
            
            response += "\n💡 Хотите узнать больше о вакансии? Напишите её номер!"
            
            await update.message.reply_text(response)
            
        except Exception as e:
            logger.error(f"Error processing resume: {e}")
            await update.message.reply_text(
                "❌ Ошибка при обработке резюме. Попробуйте еще раз."
            )
        finally:
            db.close()
    
    def build_application(self):
        """Build and configure the Telegram application"""
        self.application = Application.builder().token(self.token).build()
        
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("vacancies", self.vacancies_command))
        self.application.add_handler(CommandHandler("my_profile", self.my_profile_command))
        self.application.add_handler(CommandHandler("chat", self.chat_command))
        self.application.add_handler(CommandHandler("stop", self.stop_chat_command))
        
        self.application.add_handler(
            MessageHandler(filters.Document.PDF, self.handle_document)
        )
        
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_or_chat)
        )
        
        return self.application
    
    async def run(self):
        """Run the bot in background"""
        if not self.application:
            self.build_application()
        
        if not self.application:
            logger.error("Failed to build application")
            return
        
        logger.info("Starting HR AI Assistant Bot...")
        
        try:
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
            logger.info("Bot polling started successfully")
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            raise

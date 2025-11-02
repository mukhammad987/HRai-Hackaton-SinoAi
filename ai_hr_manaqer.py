"""
AI HR Manager - Intelligent chat assistant for employers
Uses OpenAI via Replit AI Integrations
"""

import os
from openai import OpenAI
from typing import List, Dict

# the newest OpenAI model is "gpt-5" which was released August 7, 2025.
# do not change this unless explicitly requested by the user
AI_MODEL = "gpt-5"

# This is using Replit's AI Integrations service, which provides OpenAI-compatible API access
# without requiring your own OpenAI API key. Charges are billed to Replit credits.
AI_INTEGRATIONS_OPENAI_API_KEY = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
AI_INTEGRATIONS_OPENAI_BASE_URL = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")


class AIHRManager:
    """AI HR Manager for chatting with employers about recruitment"""
    
    def __init__(self):
        """Initialize OpenAI client using Replit AI Integrations"""
        self.client = OpenAI(
            api_key=AI_INTEGRATIONS_OPENAI_API_KEY,
            base_url=AI_INTEGRATIONS_OPENAI_BASE_URL
        )
        
        self.system_prompt = """Ты - профессиональный HR менеджер AI Assistant.
        
Твоя роль:
- Помогать работодателям с вопросами о подборе персонала
- Консультировать по созданию вакансий
- Объяснять как работает AI подбор кандидатов
- Давать советы по формулировке требований к вакансиям
- Помогать анализировать результаты подбора

Ты умеешь:
✅ Создавать эффективные описания вакансий
✅ Формулировать требования к кандидатам
✅ Советовать какие навыки указать
✅ Объяснять метрики подбора (процент соответствия)
✅ Помогать понять результаты AI анализа

Отвечай:
- На русском языке
- Профессионально и дружелюбно
- Кратко и по делу
- С конкретными примерами
- Используй эмодзи для улучшения читаемости

Если работодатель спрашивает как добавить вакансию - скажи что это можно сделать:
1. Через веб-панель управления (форма "Добавить вакансию")
2. Через команду /add_vacancy в боте

Если спрашивают о кандидатах - объясни что кандидаты появляются когда соискатели отправляют резюме через бот.

Твоя цель - помочь работодателю эффективно найти нужных специалистов!"""
    
    def chat(self, user_message: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Chat with AI HR Manager
        
        Args:
            user_message: User's question or message
            chat_history: Previous conversation history (optional)
            
        Returns:
            AI response
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if chat_history:
            messages.extend(chat_history)
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=AI_MODEL,
                messages=messages,
                max_completion_tokens=1000
            )
            
            return response.choices[0].message.content or "Извините, не смог сформулировать ответ."
            
        except Exception as e:
            return f"Ошибка AI: {str(e)}"
    
    def get_vacancy_writing_help(self, job_title: str) -> str:
        """
        Get help writing a vacancy description
        
        Args:
            job_title: Title of the job position
            
        Returns:
            Suggestions for vacancy description
        """
        prompt = f"""Помоги составить вакансию для позиции: {job_title}

Предложи:
1. Краткое описание обязанностей
2. Ключевые требования к кандидату
3. Важные навыки для указания в вакансии

Формат: структурированный, готовый для копирования."""
        
        return self.chat(prompt)
    
    def analyze_match_score(self, score: float, matching_skills: List[str]) -> str:
        """
        Explain a match score to an employer
        
        Args:
            score: Match score percentage
            matching_skills: List of matching skills
            
        Returns:
            Explanation of the match
        """
        skills_text = ", ".join(matching_skills) if matching_skills else "нет совпадений"
        
        prompt = f"""Объясни работодателю результат AI подбора:

Процент соответствия: {score}%
Совпадающие навыки: {skills_text}

Дай короткую рекомендацию: стоит ли рассматривать этого кандидата и почему?"""
        
        return self.chat(prompt)


# Singleton instance
_ai_manager = None

def get_ai_manager() -> AIHRManager:
    """Get singleton AI HR Manager instance"""
    global _ai_manager
    if _ai_manager is None:
        _ai_manager = AIHRManager()
    return _ai_manager

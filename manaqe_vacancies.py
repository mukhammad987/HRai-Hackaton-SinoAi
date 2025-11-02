"""
Vacancy Management Utility
Add, update, and manage job vacancies
"""

import sys
from database import init_database, get_session, Vacancy
from matching import get_matching_service


def add_sample_vacancies():
    """Add sample vacancies to the database"""
    print("Инициализация базы данных...")
    init_database()
    
    print("Загрузка AI модели...")
    matching_service = get_matching_service()
    
    sample_vacancies = [
        {
            "title": "Senior Python Developer",
            "company": "Tech Innovations Inc.",
            "location": "Москва / Удаленно",
            "employment_type": "Полная занятость",
            "description_text": """
Мы ищем опытного Python разработчика для работы над ML проектами.
Требования: Python, FastAPI, Machine Learning, TensorFlow/PyTorch, SQL.
Опыт работы: 5+ лет в разработке.
            """,
            "requirements_text": """
- Глубокие знания Python и его экосистемы
- Опыт с ML фреймворками (TensorFlow, PyTorch, scikit-learn)
- Знание FastAPI или Django
- Опыт работы с базами данных (PostgreSQL, MongoDB)
- Понимание DevOps практик
            """,
            "tags": ["python", "machine-learning", "fastapi", "senior"]
        },
        {
            "title": "Data Scientist",
            "company": "Analytics Solutions Ltd.",
            "location": "Санкт-Петербург",
            "employment_type": "Полная занятость",
            "description_text": """
Аналитическая компания ищет Data Scientist для работы с большими данными.
Требования: Python, Pandas, NumPy, ML, статистика, визуализация данных.
Опыт: 3+ года в Data Science.
            """,
            "requirements_text": """
- Опыт работы с Python и библиотеками анализа данных
- Знание статистики и машинного обучения
- Умение создавать предиктивные модели
- Опыт с Pandas, NumPy, Matplotlib, Seaborn
- SQL и работа с базами данных
            """,
            "tags": ["data-science", "python", "analytics", "ml"]
        },
        {
            "title": "Machine Learning Engineer",
            "company": "AI Innovations Corp.",
            "location": "Удаленно",
            "employment_type": "Полная занятость",
            "description_text": """
Разработка и внедрение ML моделей в production.
Требования: Python, ML frameworks, NLP, Computer Vision, deployment.
Опыт: 4+ года в ML/AI.
            """,
            "requirements_text": """
- Опыт разработки и deployment ML моделей
- Знание NLP и компьютерного зрения
- Работа с Transformers, BERT, GPT
- Опыт с облачными платформами (AWS, Azure)
- MLOps и CI/CD для ML
            """,
            "tags": ["machine-learning", "nlp", "computer-vision", "mlops"]
        },
        {
            "title": "Full Stack Developer",
            "company": "Web Studio Pro",
            "location": "Казань",
            "employment_type": "Полная занятость",
            "description_text": """
Разработка веб-приложений full stack.
Требования: Python/JavaScript, React, FastAPI/Django, PostgreSQL.
Опыт: 3+ года в веб-разработке.
            """,
            "requirements_text": """
- Frontend: React, JavaScript/TypeScript
- Backend: Python, FastAPI или Django
- Базы данных: PostgreSQL, Redis
- Git, Docker, основы DevOps
- Опыт создания REST API
            """,
            "tags": ["fullstack", "react", "python", "web-development"]
        },
        {
            "title": "Junior Python Developer",
            "company": "StartUp Tech",
            "location": "Удаленно",
            "employment_type": "Полная занятость",
            "description_text": """
Ищем начинающего Python разработчика в растущую команду.
Требования: Python basics, SQL, желание учиться, базовые знания веб-фреймворков.
Опыт: 1+ год или выпускник профильного вуза.
            """,
            "requirements_text": """
- Базовые знания Python
- Понимание ООП и структур данных
- Знакомство с SQL
- Git и основы командной работы
- Желание развиваться в ML/Web
            """,
            "tags": ["python", "junior", "entry-level", "remote"]
        }
    ]
    
    db = get_session()
    try:
        print(f"\nДобавление {len(sample_vacancies)} вакансий...\n")
        
        for i, vac_data in enumerate(sample_vacancies, 1):
            full_text = vac_data['description_text']
            if vac_data.get('requirements_text'):
                full_text += "\n" + vac_data['requirements_text']
            
            print(f"{i}. Создание вакансии: {vac_data['title']}")
            print(f"   Генерация embeddings...")
            
            embedding = matching_service.get_embedding(full_text)
            
            vacancy = Vacancy(
                title=vac_data['title'],
                company=vac_data['company'],
                location=vac_data['location'],
                employment_type=vac_data['employment_type'],
                description_text=vac_data['description_text'],
                requirements_text=vac_data.get('requirements_text'),
                tags=vac_data.get('tags', []),
                is_active=True
            )
            vacancy.set_embedding(embedding)
            
            db.add(vacancy)
            print(f"   ✅ Добавлено\n")
        
        db.commit()
        print(f"✅ Все вакансии успешно добавлены в базу данных!\n")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        db.rollback()
    finally:
        db.close()


def list_vacancies():
    """List all vacancies in database"""
    db = get_session()
    try:
        vacancies = db.query(Vacancy).all()
        
        if not vacancies:
            print("Вакансий в базе данных нет.")
            return
        
        print(f"\nВсего вакансий: {len(vacancies)}\n")
        print("=" * 80)
        
        for vac in vacancies:
            status = "✅ Активна" if vac.is_active else "❌ Неактивна"
            print(f"\nID: {vac.id}")
            print(f"Название: {vac.title}")
            print(f"Компания: {vac.company}")
            print(f"Локация: {vac.location}")
            print(f"Статус: {status}")
            print(f"Создана: {vac.created_at.strftime('%d.%m.%Y %H:%M')}")
            print("-" * 80)
        
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "add":
            add_sample_vacancies()
        elif command == "list":
            list_vacancies()
        else:
            print("Неизвестная команда. Используйте: add или list")
    else:
        print("Утилита управления вакансиями")
        print("\nИспользование:")
        print("  python manage_vacancies.py add   - Добавить примеры вакансий")
        print("  python manage_vacancies.py list  - Показать все вакансии")

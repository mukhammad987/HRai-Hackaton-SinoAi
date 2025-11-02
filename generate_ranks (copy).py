# generate_ranks.py — СОЗДАЁТ CSV ДЛЯ СДАЧИ НА ХАКАТОН

import os
import pandas as pd
from improved_matching import get_improved_matching_service
from pdf_parser import parse_pdf_file

# === НАСТРОЙКИ (измени на свои папки!) ===
VACANCY_DIR = "vacancies/"      # Папка с PDF-вакансиями
RESUME_DIR = "resumes/"         # Папка с PDF-резюме
OUTPUT_CSV = "ranks.csv"        # Имя итогового файла

# === ЗАГРУЖАЕМ МОДЕЛЬ ===
print("Загружаю модель для подбора...")
service = get_improved_matching_service()
print("Готово!")

# === СОБИРАЕМ ФАЙЛЫ ===
vacancy_files = [f for f in os.listdir(VACANCY_DIR) if f.endswith(".pdf")]
resume_files = [f for f in os.listdir(RESUME_DIR) if f.endswith(".pdf")]

print(f"Найдено вакансий: {len(vacancy_files)}")
print(f"Найдено резюме: {len(resume_files)}")

# === СЧИТАЕМ РАНГИ ===
rows = []

for vac_file in vacancy_files:
    vac_path = os.path.join(VACANCY_DIR, vac_file)
    print(f"\nОбрабатываю вакансию: {vac_file}")

    vac_text = parse_pdf_file(vac_path)

    scores = []
    for res_file in resume_files:
        res_path = os.path.join(RESUME_DIR, res_file)
        res_text = parse_pdf_file(res_path)

        # Твой matching!
        result = service.match_resume_to_vacancy(res_text, vac_text)
        scores.append((res_file, result['score']))

    # Сортируем: от лучшего к худшему
    scores.sort(key=lambda x: x[1], reverse=True)

    # Записываем топ-ранги
    for rank, (res_file, score) in enumerate(scores, 1):
        rows.append({
            "job_file": vac_file,
            "rank": rank,
            "resume_file": res_file
        })
        print(f"  {rank}. {res_file} → {score:.1f}%")

# === СОХРАНЯЕМ В CSV ===
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nCSV готов! Сохранён как: {OUTPUT_CSV}")
print("Загрузи этот файл на GitHub — и сдавай проект!")

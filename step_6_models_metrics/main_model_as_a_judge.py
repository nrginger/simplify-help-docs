import os
import time
import pandas as pd
import requests
import json
from dotenv import load_dotenv
from tqdm import tqdm

# Загрузка переменных окружения
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Конфигурация Claude 3.7 Sonnet через aitunnel
BASE_URL = 'https://api.aitunnel.ru/v1/chat/completions'
HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_KEY}'
}

# 1. Читаем и случайно выбираем 50 строк
sample_df = pd.read_csv('combined_sample50_round2.csv')

# 2. Подготавливаем список моделей и критериев
models = ['llama_finetuned', 'mistral_instruct', 'qwen_finetuned', 'deepseek-v3']
criteria = [
    "clarity",  # ясность и понятность
    "direct_address",  # обращение к пользователю
    "no_jargon",  # отсутствие канцеляризмов
    "structure",  # логичность структуры
    "brevity"  # краткость
]


# 3. Функция запроса к Claude
def judge_style(original, candidates):
    prompt = (
        "Вы — высококлассный редактор и технический писатель. "
        "Оцени следующие тексты: они должны соответствовать инфостилю "
        "и лучшим практикам документирования Microsoft, Google и Apple "
        "только для русского языка.\n\n"
        "Оцени каждый упрощённый текст по 5 критериям по шкале от 1 до 5:\n"
        "1. clarity — ясность и понятность: насколько текст легко читается.\n"
        "2. direct_address — обращение к пользователю: использование «вы».\n"
        "3. bureaucratic_words — отсутствие канцеляризмов и клише.\n"
        "4. structure — логичность структуры и связность.\n"
        "5. brevity — краткость без потери важной информации.\n\n"
        "Не добавляй комментариев — проставляй только числа.\n\n"
        f"Оригинал:\n{original}\n\n"
    )
    for name, text in candidates.items():
        prompt += f"{name}:\n{text}\n\n"
    prompt += (
        "Ответ — только JSON, никаких пояснений и кавычек вне структуры. "
        "Если не можешь вернуть JSON — верни {\"error\":\"no_json\"}.\n"
        "{\n"
        + ",\n".join(
            f'  "{name}": {{'
            '"clarity":0, '
            '"direct_address":0, '
            '"bureaucratic_words":0, '
            '"structure":0, '
            '"brevity":0'
            "}}"
            for name in candidates.keys()
        )
        + "\n}\n"
    )

    payload = {
        "model": "claude-3.7-sonnet",
        "messages": [{"role":"user","content": prompt}],
        "temperature": 0.0,
        "max_tokens": 800
    }
    resp = requests.post(BASE_URL, headers=HEADERS, json=payload)
    resp.raise_for_status()

    # Попытки распарсить JSON
    for attempt in range(2):
        content = resp.json()['choices'][0]['message']['content'].strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print(f"❗ Некорректный JSON (попытка {attempt + 1}):", content)
            time.sleep(1)
            resp = requests.post(BASE_URL, headers=HEADERS, json=payload)

    # Если не получилось после двух попыток — возвращаем пустые оценки
    return {name: {crit: None for crit in criteria} for name in candidates}


# 4. Прогоняем sample_df
records = []
for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Judging"):
    orig = row['original_text']
    candidates = {m: row[m] for m in models}
    try:
        scores = judge_style(orig, candidates)
    except Exception as e:
        print(f"Error on row {_}: {e}")
        scores = {m: {crit: None for crit in criteria} for m in models}
    # Разворачиваем в плоскую запись
    rec = {}
    for m, crits in scores.items():
        for crit, val in crits.items():
            rec[f"{m}_{crit}"] = val
    records.append(rec)
    time.sleep(5)

# 5. Сохраняем результаты
scores_df = pd.DataFrame(records)
output = pd.concat([sample_df, scores_df], axis=1)
output.to_csv('combined_sample50_judge_round2.csv', index=False)

import json
with open('combined_sample50_judge_round2.json', 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=2)


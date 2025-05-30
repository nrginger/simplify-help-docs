import pandas as pd
import time
import os
import logging
import random
from tqdm import tqdm
from dotenv import load_dotenv
import openai

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simplify_texts.log"), logging.StreamHandler()
    ]
)

# Загрузка переменных окружения
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Проверка наличия API ключа
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY не найден в переменных окружения.")

# Константы
INPUT_FILE = "unique_texts_200_formatted_golden_standard.xlsx"
OUTPUT_FILE = "unique_texts_200_formatted_golden_standard_simplified.xlsx"
TEMP_FILE = "unique_texts_200_formatted_golden_standard_temp.xlsx"
BATCH_SAVE_SIZE = 5  # Сохранять каждые 5 обработанных текстов
MIN_PAUSE = 1  # Минимальная пауза между запросами в секундах
MAX_PAUSE = 3  # Максимальная пауза между запросами в секундах
OUTPUT_COLUMN = "deepseek-v3"  # Название выходного столбца

# Промпт для упрощения текста
SIMPLIFY_PROMPT = """Ты профессиональный редактор, который делает сложные тексты простыми и понятными.
Следуй этим правилам строго:
1. Перепиши сложный технический текст простыми, понятными словами.
2. Сохрани все важные факты и смысл оригинала.
3. Используй короткие предложения.
4. Избегай сложных терминов или объясняй их.
5. Не пиши вводных фраз типа "В этом тексте", "Этот текст о", "Безусловно", "Можно сказать".
6. Не пиши о процессе упрощения.
7. Не добавляй свои мнения или оценки.
8. Начинай сразу с упрощенного текста без вступлений.

Текст для упрощения: {text}"""


def get_simplified_text(text, retry_count=3):
    """
    Получает упрощенную версию текста от DeepSeek API через прокси aitunnel.ru
    """
    messages = [
        {"role": "system",
         "content": "Ты профессиональный редактор, который делает сложные тексты простыми и понятными."},
        {"role": "user", "content": SIMPLIFY_PROMPT.format(text=text)}
    ]

    for attempt in range(retry_count):
        try:
            logging.info(f"Отправка запроса к API (попытка {attempt + 1}/{retry_count})")

            client = openai.OpenAI(
                api_key=OPENAI_API_KEY,
                base_url="https://api.aitunnel.ru/v1/"
            )

            response = client.chat.completions.create(
                model="deepseek-chat-v3-0324",
                messages=messages,
                temperature=0.5,
                max_tokens=2000,
                top_p=0.9
            )

            if response and response.choices:
                result = response.choices[0].message.content
                logging.info(f"Успешно получен ответ от API (длина: {len(result)})")
                return result.strip()
            else:
                logging.warning(f"Попытка {attempt + 1}: Пустой ответ")

        except Exception as e:
            logging.error(f"Попытка {attempt + 1}/{retry_count}: Ошибка API - {str(e)}")
            if attempt < retry_count - 1:  # Если это не последняя попытка
                pause = 5 + (attempt * 2)  # Увеличиваем паузу с каждой попыткой
                logging.info(f"Пауза перед следующей попыткой: {pause} секунд")
                time.sleep(pause)

    logging.error(f"Не удалось получить ответ после {retry_count} попыток")
    return f"ОШИБКА: Не удалось обработать текст после {retry_count} попыток."


def main():
    # Проверка существования входного файла
    if not os.path.exists(INPUT_FILE):
        logging.error(f"Файл {INPUT_FILE} не найден!")
        return

    # Загрузка данных
    try:
        df = pd.read_excel(INPUT_FILE)
        logging.info(f"Загружено {len(df)} текстов из {INPUT_FILE}")
    except Exception as e:
        logging.error(f"Ошибка при загрузке файла {INPUT_FILE}: {e}")
        return

    # Проверка наличия первого столбца
    if len(df.columns) == 0:
        logging.error("Файл не содержит столбцов!")
        return

        # Получаем название первого столбца
    first_column = df.columns[0]
    logging.info(f"Используем столбец '{first_column}' как источник текстов")

    # Добавляем столбец для упрощенных текстов, если его нет
    if OUTPUT_COLUMN not in df.columns:
        df[OUTPUT_COLUMN] = None
        logging.info(f"Добавлен новый столбец '{OUTPUT_COLUMN}' для упрощенных текстов")

    # Подсчет текстов, которые нужно обработать
    texts_to_process = df[df[OUTPUT_COLUMN].isna()].shape[0]
    logging.info(f"Необходимо обработать {texts_to_process} текстов")

    # Обработка текстов
    processed_count = 0

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Упрощение текстов"):
        # Пропускаем, если текст уже обработан
        if not pd.isna(df.at[index, OUTPUT_COLUMN]):
            continue

        # Получаем оригинальный текст
        original_text = str(row[first_column])
        logging.info(f"Обработка текста #{index}: {original_text[:100]}...")

        # Получаем упрощенную версию
        simplified_text = get_simplified_text(original_text)

        # Сохраняем результат
        if simplified_text:
            df.at[index, OUTPUT_COLUMN] = simplified_text
            processed_count += 1
            logging.info(f"Текст #{index} успешно упрощен. Результат: {simplified_text[:100]}...")
        else:
            logging.error(f"Не удалось упростить текст #{index}")
            df.at[index, OUTPUT_COLUMN] = "ОШИБКА: Не удалось обработать текст."

        # Сохраняем промежуточные результаты
        if processed_count % BATCH_SAVE_SIZE == 0:
            logging.info(f"Сохранение промежуточных результатов после {processed_count} обработанных текстов...")
            df.to_excel(TEMP_FILE, index=False)

        # Пауза между запросами
        pause_time = random.uniform(MIN_PAUSE, MAX_PAUSE)
        logging.info(f"Пауза {pause_time:.2f} секунд...")
        time.sleep(pause_time)

    # Сохранение финального результата
    try:
        df.to_excel(OUTPUT_FILE, index=False)
        logging.info(f"Результаты сохранены в {OUTPUT_FILE}")

        # Удаляем временный файл, если операция успешна
        if os.path.exists(TEMP_FILE):
            os.remove(TEMP_FILE)
            logging.info(f"Временный файл {TEMP_FILE} удален")
    except Exception as e:
        logging.error(f"Ошибка при сохранении результатов: {e}")
        logging.info(f"Пожалуйста, используйте временный файл {TEMP_FILE} для восстановления данных")


if __name__ == "__main__":
    logging.info("Запуск программы упрощения текстов...")
    try:
        main()
        logging.info("Программа завершена успешно!")
    except Exception as e:
        logging.error(f"Программа завершена с ошибкой: {e}")
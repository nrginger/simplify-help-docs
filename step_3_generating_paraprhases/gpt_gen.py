import pandas as pd
import requests
import time
import os
import logging
import random
import json
from tqdm import tqdm
from dotenv import load_dotenv

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("paraphrase.log"),
        logging.StreamHandler()
    ]
)

# Загрузка переменных окружения
load_dotenv()
CAILA_TOKEN = os.getenv('CAILA_TOKEN')

# Проверка наличия токена
if not CAILA_TOKEN:
    raise ValueError("CAILA_TOKEN не найден в переменных окружения.")

# Константы
INPUT_FILE = "gpt_3.csv"
OUTPUT_FILE = "gpt_3_paraphrased.csv"
TEMP_FILE = "gpt_3_paraphrased_temp.csv"
BATCH_SAVE_SIZE = 10  # Сохранять каждые 10 обработанных текстов
MIN_PAUSE = 2  # Минимальная пауза между запросами в секундах
MAX_PAUSE = 5  # Максимальная пауза между запросами в секундах
PROMPT_FREQUENCY = 2  # Частота повторного включения инструкции (каждые 2 текста)

# Промпт для перефразирования
PARAPHRASE_PROMPT = """ 
Ты - опытный технический писатель. Твоя специализация — улучшение пользовательской документации, руководств пользователя. Ты придерживаешься легкого инфостиля - информационного стиля. Тебя вдохновляют взгляды и работы Максима Ильяхова и Норы Галь - книга "Слово живое и мертвое". Выполняй правила ниже.
Попутно исправляй ошибки, которые возникли в результате распознавания PDF в TXT.

##Правила##
1. Делай:
   - Переписывай текст так, чтобы он стал простым и легким для восприятия.
   - Сохраняй технические термины и названия брендов без изменений.
   - Разделяй большие предложения на более короткие и четкие.
   - Заменяй причастия на конструкции "который + глагол", например "параметры, сохраняемые в памяти" --> "параметры, которые сохраняются в памяти".
   - Переделывай деепричастные обороты в новое предложение.
   - Обращайся к пользователю напрямую, используя "вы".
   - Применяй активный залог и добавляй больше глаголов для ясности.
   - Где возможно – заменяй существительные глаголами.
   - Соблюдай доброжелательный и поддерживающий tone of voice.
   - Если встречаешь символы, которые не имеют значения и не подходят по контексту – удаляй их.
   - Если видишь пробелы между словами, которых быть не должно – убирай пробелы.
   - Если в словах есть дефисы, которых быть не должно – убирай дефисы.
2. НЕ делай:
   - НЕ используй канцеляризмы, клише и манипулятивные фразы.
   - НЕ используй причастные и деепричастные обороты.
   - НЕ используй скобки и сложные синтаксические конструкции.
   - НЕ используй модальные слова, такие как "нужно", "можно", "необходимо", "следует" и другие подобные слова.
   - НЕ используй слова "данный" и "следующий", заменяй на конкретные указательные местоимения, например "этот", если они нужны.
   - НЕ используй слово "является", заменяй его на тире.
   - НЕ используй пассивные формы — выбирай активные и прямые конструкции.

##КОНТЕКСТ##
Тебе даны отрывки технической документации, которые нужно переработать в легкий и понятный стиль. Этот стиль должен соответствовать стандартам документации таких компаний, как Яндекс и Google. Тексты должны быть ясными и четкими, оставаясь в рамках технического домена, без перехода в разговорный стиль. Образцы переписанных фраз:
- Пример: "для настройки питания блока" --> "чтобы настроить питание блока".
- Пример: "Регулировка и настройка двигателя" --> "Как отрегулировать и настроить двигатель"
- Пример: "Выполните это для каждого устройства, изменяя положение переключателей согласно таблице." --> "Выполните это для каждого устройства. Изменяйте положение переключателей по таблице."
- Пример: "деталь является частью прибора" --> "деталь - это часть прибора"
- Пример: "связь, устанавливаемая" --> "связь, которая устанавливается"
- Пример: "при наличии заслонки" --> "если есть заслонка"
- Пример: "при эксплуатации холодильника" --> "когд вы используете холодильник"
- Пример: "производить проверку" --> "проверять"
- Пример: "осуществлять контроль" --> "контролировать"
- Пример: "выполнять работу" --> "работать"
- Пример: "совершать покупку" --> "покупать"
- Пример: "до момента открытия" --> "до того, как откроется"
Также учитывай, что инструкции были распознаны из PDF, поэтому они могут содержать артефакты: лишние символы, знаки препинания, неправильно распознанные слова. Исправляй такие недочеты.
Следуй этим инструкциям для достижения наиболее точного и эффективного результата.

ВАЖНО: Если предложение написано достаточно хорошо, оставляй его без изменений. 

Перепиши этот текст:

Текст: {text}

Переписанный текст:"""


def get_gpt_response(text, retry_count=3, use_full_prompt=True):
    """
    Получает ответ от GPT-o4-mini через прокси.

    Args:
        text: Текст для перефразирования
        retry_count: Количество попыток в случае ошибки
        use_full_prompt: Использовать полный промпт с инструкцией или только текст

    Returns:
        Перефразированный текст или None в случае ошибки
    """
    headers = {
        "Authorization": f"Bearer {CAILA_TOKEN}",
        "MLP-API-KEY": CAILA_TOKEN,
        "Content-Type": "application/json"
    }

    # Формируем содержимое запроса в зависимости от use_full_prompt
    if use_full_prompt:
        content = PARAPHRASE_PROMPT.format(text=text)
    else:
        content = f"Перефразируй: {text}"

    messages = [
        {"role": "system", "content": "Ты - опытный технический писатель, который умеет хорошо переписывать и исправлять пользовательскую документацию, мануалы, руководства по эксплуатации, сохраняя их смысл."},
        {"role": "user", "content": content}
    ]

    payload = {
        "chat": {
            "model": "gpt-4o-mini",  # Модель GPT-o4-mini
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500
        }
    }

    for attempt in range(retry_count):
        try:
            response = requests.post(
                "https://caila.io/api/mlpgate/account/just-ai/model/openai-proxy/predict",
                headers=headers,
                json=payload,
                timeout=30  # Таймаут 30 секунд
            )
            response.raise_for_status()

            response_data = response.json()
            if ('chat' in response_data and
                    'choices' in response_data['chat'] and
                    len(response_data['chat']['choices']) > 0 and
                    'message' in response_data['chat']['choices'][0]):
                return response_data['chat']['choices'][0]['message']['content']
            else:
                logging.warning(f"Неожиданный формат ответа: {json.dumps(response_data)}")
                if attempt < retry_count - 1:
                    time.sleep(5)  # Пауза перед повторной попыткой
                continue

        except Exception as e:
            logging.error(f"Попытка {attempt + 1}/{retry_count}: Ошибка при получении ответа от GPT: {e}")
            if attempt < retry_count - 1:
                time.sleep(5)  # Пауза перед повторной попыткой

        return None  # Все попытки неудачны


def main():
    # Проверка существования входного файла
    if not os.path.exists(INPUT_FILE):
        logging.error(f"Файл {INPUT_FILE} не найден!")
        return

    # Загрузка данных
    try:
        df = pd.read_csv(INPUT_FILE)
        logging.info(f"Загружено {len(df)} текстов из {INPUT_FILE}")
    except Exception as e:
        logging.error(f"Ошибка при загрузке файла {INPUT_FILE}: {e}")
        return

    # Создание столбца для перефразированных текстов
    if 'text' not in df.columns:
        logging.error(f"Столбец 'text' не найден в {INPUT_FILE}")
        return

    # Проверка наличия временного файла для продолжения работы
    if os.path.exists(TEMP_FILE):
        try:
            temp_df = pd.read_csv(TEMP_FILE)
            processed_count = len(temp_df)
            logging.info(f"Найден временный файл с {processed_count} обработанными текстами. Продолжаем с этого места.")

            # Объединение уже обработанных данных с оставшимися
            processed_texts = set(temp_df['original_text'])
            remaining_df = df[~df['text'].isin(processed_texts)]

            # Обновляем df для обработки оставшихся текстов
            df = remaining_df.copy()
            result_df = temp_df.copy()
        except Exception as e:
            logging.error(f"Ошибка при загрузке временного файла: {e}")
            result_df = pd.DataFrame(columns=['original_text', 'paraphrased_text'])
    else:
        result_df = pd.DataFrame(columns=['original_text', 'paraphrased_text'])

    # Обработка текстов
    processed_count = 0
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Перефразирование текстов"):
        original_text = row['text']

        # Определение, использовать ли полный промпт с инструкцией
        use_full_prompt = (processed_count % PROMPT_FREQUENCY == 0)

        # Получение перефразированного текста
        paraphrased_text = get_gpt_response(original_text, use_full_prompt=use_full_prompt)

        if paraphrased_text:
            # Добавление результата
            result_df = pd.concat([result_df, pd.DataFrame({
                'original_text': [original_text],
                'paraphrased_text': [paraphrased_text]
            })], ignore_index=True)

            processed_count += 1

            # Сохранение промежуточных результатов
            if processed_count % BATCH_SAVE_SIZE == 0:
                result_df.to_csv(TEMP_FILE, index=False)
                logging.info(f"Промежуточное сохранение: обработано {processed_count} текстов")
        else:
            logging.warning(f"Не удалось перефразировать текст: {original_text[:50]}...")

        # Пауза между запросами
        pause_time = random.uniform(MIN_PAUSE, MAX_PAUSE)
        time.sleep(pause_time)

    # Сохранение финального результата
    if not result_df.empty:
        result_df.to_csv(OUTPUT_FILE, index=False)
        logging.info(f"Готово! Обработано {len(result_df)} текстов. Результат сохранен в {OUTPUT_FILE}")

        # Удаление временного файла при успешном завершении
        if os.path.exists(TEMP_FILE):
            os.remove(TEMP_FILE)
    else:
        logging.warning("Не удалось обработать ни один текст!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Процесс прерван пользователем")
    except Exception as e:
        logging.error(f"Неожиданная ошибка: {e}", exc_info=True)

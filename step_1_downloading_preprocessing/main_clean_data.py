import pandas as pd
import re
import string
import os

# Путь к файлу
input_path = r"D:\docs_dataset_new\text_blocks_2.xlsx"
output_path = r"D:\docs_dataset_new\text_blocks_2_cleaned.xlsx"

# Загружаем датасет
print("Загрузка датасета...")
df = pd.read_excel(input_path)
initial_rows = len(df)
print(f"Изначальное количество строк: {initial_rows}")


# Функция для очистки текста
def clean_text(text):
    if not isinstance(text, str):
        return None

    # Замена часто встречающихся артефактов распознавания PDF
    replacements = {
        '•': '-',  # Bullet points на дефисы
        '″': '"',  # Кавычки
        '′': "'",  # Апострофы
        '–': '-',  # Тире на дефис
        '—': '-',  # Длинное тире на дефис
        '\xa0': ' ',  # Неразрывный пробел
        '…': '...',  # Многоточие
        '≥': '>=',  # Больше или равно
        '≤': '<=',  # Меньше или равно
        '©': '(c)',  # Копирайт
        '®': '(R)',  # Зарегистрированная марка
        '\t': ' ',  # Табуляция
        '\f': ' ',  # Разрыв страницы
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Исправление распространенных ошибок распознавания
    text = re.sub(r'([а-яА-Яa-zA-Z])\s+([.,;:?!])', r'\1\2', text)  # Убираем пробелы перед знаками препинания
    text = re.sub(r'\s+', ' ', text)  # Заменяем множественные пробелы на один
    text = re.sub(r'([.!?])\s*([А-ЯA-Z])', r'\1 \2', text)  # Добавляем пробел после знаков конца предложения

    # Исправление типичных OCR ошибок
    text = re.sub(r'l([^a-zA-Z])', r'1\1', text)  # l на 1 если не в слове
    text = re.sub(r'O([^a-zA-Z])', r'0\1', text)  # O на 0 если не в слове

    # Удаление последовательностей символов не на кириллице длиной более 50 знаков
    text = re.sub(r'[^а-яА-ЯёЁ\s.,!?:;()\-\"\'0-9]{50,}', '', text)

    # Удаление лишних символов переноса
    text = re.sub(r'-\s+', '', text)

    # Очистка от странных знаков
    text = ''.join([c if c in string.printable or c.isalpha() else ' ' for c in text])

    # Финальная нормализация пробелов
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Критерии для удаления строк
def should_remove(text):
    if not isinstance(text, str):
        return True

    # Очищаем для проверки
    cleaned_text = clean_text(text)

    if cleaned_text is None or len(cleaned_text) == 0:
        return True

    # Удалять короткие фрагменты (менее 250 символов)
    if len(cleaned_text) < 250:
        return True

    # Удалять строки с низким соотношением русских/английских букв к общему числу символов
    alpha_count = sum(1 for c in cleaned_text if c.isalpha())
    if alpha_count < len(cleaned_text) * 0.5:
        return True

    # Удалять строки с большим количеством нераспознанных слов
    # (признак: 3 или более ? подряд или много необычных символов)
    if '???' in cleaned_text or re.search(r'[\?\[\]\{\}]{3,}', cleaned_text):
        return True

    # Проверка на наличие большого процента "мусорных" символов
    garbage_chars = sum(1 for c in cleaned_text if not c.isalnum() and not c.isspace() and c not in string.punctuation)
    if garbage_chars > len(cleaned_text) * 0.15:  # Если более 15% - мусорные символы
        return True

    # Проверка на страницы, номера, заголовки, содержащие только цифры и т.п.
    if re.match(r'^[0-9\s\-\.]+$', cleaned_text):
        return True

    # Проверка на строки с колонтитулами и техническими пометками
    if re.search(r'(страница|стр\.|page|колонтитул)', cleaned_text.lower()):
        return True

    return False


# Применяем очистку
print("Очистка текста...")
df['cleaned_text'] = df.iloc[:, 1].apply(clean_text)

# Определяем, какие строки следует удалить
print("Определение строк для удаления...")
rows_to_remove = df['cleaned_text'].apply(should_remove)
print(f"Количество строк для удаления: {rows_to_remove.sum()}")

# Удаляем строки
df = df[~rows_to_remove]
print(f"Строк после очистки: {len(df)}")

# Заменяем исходный текст на очищенный
df.iloc[:, 1] = df['cleaned_text']
df = df.drop(columns=['cleaned_text'])

# Сохраняем очищенный датасет
print(f"Сохранение очищенного датасета в {output_path}...")
df.to_excel(output_path, index=False)

print(f"Готово! Удалено {initial_rows - len(df)} строк.")
print(f"Итоговый размер датасета: {len(df)} строк.")

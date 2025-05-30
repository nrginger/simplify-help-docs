import os
import re
import nltk
import langid
import pandas as pd

# Загрузка необходимых ресурсов NLTK
nltk.download('punkt')
nltk.download('stopwords')

def is_likely_header(line):
    """
    Определяет, является ли строка вероятным заголовком.
    Возвращает True, если строка подходит под критерии заголовка.
    """
    line = line.strip()

    # Пустые строки не являются заголовками
    if not line:
        return False

    # Проверка количества слов (заголовки обычно короткие)
    words = re.findall(r'\b\w+\b', line)
    if len(words) > 8 or len(words) < 1:  # слишком длинные или пустые строки - не заголовки
        return False

    # Заголовки часто не заканчиваются знаками препинания
    if line[-1] in '.,:;!?':
        return False

    # Проверка на нумерацию разделов (например, "2. " или "3.4. " и т.п.)
    has_numbering = bool(re.match(r'^(\d+\.)+\s', line) or re.match(r'^\d+\.?\s', line))

    # Для русских заголовков типична структура: первая буква заглавная, остальные строчные
    first_word = words[0] if words else ""
    is_russian_style_title = False
    if first_word and len(first_word) > 1:
        is_russian_style_title = (first_word[0].isupper() and first_word[1:].islower())

    # Заголовки могут быть полностью в верхнем регистре (например, "СОДЕРЖАНИЕ")
    all_caps = line.isupper()

    # Возвращаем True, если строка соответствует характеристикам заголовка
    return has_numbering or all_caps or is_russian_style_title

def is_russian(text):
    """
    Проверяет, написан ли текст на русском языке, используя langid.
    Возвращает True, если язык текста русский.
    """
    if not text or len(text.strip()) < 40:  # Требуется достаточное количество текста
        return False

    lang, _ = langid.classify(text)
    return lang == 'ru'

def split_into_sentences(text):
    """
    Разделяет текст на предложения, используя nltk.sent_tokenize.
    """
    sentences = nltk.sent_tokenize(text)
    return sentences

def extract_text_blocks(directory_path, output_file, min_sents=10, max_sents=30):
    """
    Извлекает блоки текста из всех .txt файлов в указанной директории.
    Сохраняет результат в CSV с именем output_file.
    """
    data = []
    processed_files = 0  # Счётчик обработанных файлов

    for filename in os.listdir(directory_path):
        if not filename.endswith('.txt'):
            continue

        file_path = os.path.join(directory_path, filename)

        # Попытка прочитать файл в корректной кодировке
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='cp1251') as file:
                    lines = file.readlines()
            except UnicodeDecodeError:
                print(f"Не удалось прочитать файл {filename} из-за проблем с кодировкой.")
                continue

        header = None
        text_block = []

        for line in lines:
            line = line.strip()

            # Пропускаем пустые строки
            if not line:
                continue

            # Проверяем, является ли строка заголовком
            if is_likely_header(line):
                # Если у нас уже есть «предыдущий» заголовок и накопленный текст, завершаем блок
                if header and text_block:
                    full_text = ' '.join(text_block)
                    if is_russian(full_text):  # Берём только русский текст
                        sentences = split_into_sentences(full_text)
                        # Нужно минимум min_sents предложений
                        if len(sentences) >= min_sents:
                            # Берём не более max_sents
                            selected_sentences = sentences[:max_sents]
                            text_with_header = f"{header}\n\n" + ' '.join(selected_sentences)
                            data.append({
                                'filename': filename,
                                'text_block': text_with_header,
                                'header': header,
                                'sentence_count': len(selected_sentences)
                            })
                # Начинаем новый блок с текущим заголовком
                header = line
                text_block = []
            else:
                # Продолжаем наполнять текстовый блок только если есть заголовок
                if header:
                    text_block.append(line)

        # Обрабатываем последний блок файла
        if header and text_block:
            full_text = ' '.join(text_block)
            if is_russian(full_text):
                sentences = split_into_sentences(full_text)
                if len(sentences) >= min_sents:
                    selected_sentences = sentences[:max_sents]
                    text_with_header = f"{header}\n\n" + ' '.join(selected_sentences)
                    data.append({
                        'filename': filename,
                        'text_block': text_with_header,
                        'header': header,
                        'sentence_count': len(selected_sentences)
                    })

        processed_files += 1
        # Промежуточный вывод каждые 100 файлов
        if processed_files % 100 == 0:
            print(f"Обработано {processed_files} файлов из ~9812")

    # Создаём DataFrame с нужной структурой
    df = pd.DataFrame(data)

    # Переупорядочиваем столбцы так, чтобы text_block был вторым
    columns_order = ['filename', 'text_block', 'header', 'sentence_count']
    df = df[columns_order]

    # Сохраняем в CSV с индексом (block_id)
    df.to_csv(output_file, index=True, index_label='block_id', encoding='utf-8')

    print(f"Обработано {processed_files} файлов, извлечено {len(data)} текстовых блоков.")
    print(f"Результаты сохранены в {output_file}.")

# Пример вызова на практике:
extract_text_blocks('D:/docs_dataset_new/all_combined_txt', 'extracted_text_blocks_2.xlsx')

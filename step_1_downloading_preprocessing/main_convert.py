import os
import fitz  # PyMuPDF
from pathlib import Path


def convert_pdf_to_txt(pdf_path, txt_path):
    """
    Конвертирует PDF файл в TXT файл используя PyMuPDF
    :param pdf_path: путь к PDF файлу
    :param txt_path: путь для сохранения TXT файла
    """
    try:
        # Открываем PDF файл
        doc = fitz.open(pdf_path)
        text = ""

        # Извлекаем текст из всех страниц
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text() + "\n"

        # Закрываем документ
        doc.close()

        # Записываем текст в TXT файл
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)

        print(f"Успешно конвертирован: {pdf_path} -> {txt_path}")
        return True
    except Exception as e:
        print(f"Ошибка при конвертации {pdf_path}: {str(e)}")
        return False


def main():
    # Исходная директория с PDF файлами
    source_dir = r"D:\docs_dataset_new\txtdoc\files"
    # Директория для сохранения TXT файлов
    target_dir = r"D:\docs_dataset_new\txtdoc\txt"

    # Создаем директорию для TXT файлов, если она не существует
    os.makedirs(target_dir, exist_ok=True)

    # Счетчики для статистики
    total_files = 0
    converted_files = 0

    # Рекурсивно обходим все файлы и папки
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                total_files += 1

                # Получаем путь к исходному PDF файлу
                pdf_path = os.path.join(root, file)

                # Создаем относительный путь относительно source_dir
                rel_path = os.path.relpath(root, source_dir)
                # Если мы находимся в корневой директории, то rel_path будет "."
                if rel_path == ".":
                    rel_path = ""

                # Создаем путь для сохранения TXT файла, сохраняя структуру папок
                txt_dir = os.path.join(target_dir, rel_path)
                os.makedirs(txt_dir, exist_ok=True)

                # Создаем имя TXT файла, заменяя расширение .pdf на .txt
                txt_filename = os.path.splitext(file)[0] + '.txt'
                txt_path = os.path.join(txt_dir, txt_filename)

                # Конвертируем PDF в TXT
                if convert_pdf_to_txt(pdf_path, txt_path):
                    converted_files += 1

    print(f"\nОбработка завершена:")
    print(f"Всего PDF файлов: {total_files}")
    print(f"Успешно конвертировано: {converted_files}")
    print(f"Не удалось конвертировать: {total_files - converted_files}")


if __name__ == "__main__":
    main()
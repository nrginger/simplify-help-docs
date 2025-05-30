import requests
from bs4 import BeautifulSoup
import os
import random
import zipfile
import time
import json
from urllib.parse import urljoin
from tqdm import tqdm

BASE_URL = "https://txtdoc.ru"
BASE_DIR = "D:\\docs_dataset_new\\txtdoc"
OUTPUT_DIR = os.path.join(BASE_DIR, "files")
STATE_FILE = os.path.join(BASE_DIR, "download_state.json")
MAX_FILES = 10000
MAX_BRANDS_PER_CATEGORY = 50
MAX_MODELS_PER_BRAND = 30

DELAY_BETWEEN_REQUESTS = 1

MAIN_CATEGORIES = [
    {"url": "https://txtdoc.ru/home_appliances/", "name": "Бытовая техника"},
    {"url": "https://txtdoc.ru/computers/", "name": "Компьютеры"},
    {"url": "https://txtdoc.ru/beauty_and_health/", "name": "Красота и здоровье"},
    {"url": "https://txtdoc.ru/musical_equipment/", "name": "Музыкальное оборудование"},
    {"url": "https://txtdoc.ru/equipment/", "name": "Оборудование"},
    {"url": "https://txtdoc.ru/communications/", "name": "Телефоны и связь"},
    {"url": "https://txtdoc.ru/transport/", "name": "Транспорт"},
    {"url": "https://txtdoc.ru/electronics/", "name": "Электроника"},
]

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

downloaded_files_list = []
if os.path.exists(STATE_FILE):
    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
            downloaded_files_list = state_data.get('downloaded_files', [])
            print(f"Загружено состояние: уже скачано {len(downloaded_files_list)} файлов")
    except Exception as e:
        print(f"Ошибка загрузки состояния: {e}")
        downloaded_files_list = []

downloaded_files = len(downloaded_files_list)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def save_state():
    state = {
        'downloaded_files': downloaded_files_list,
        'total_downloaded': downloaded_files
    }
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"Состояние сохранено: {downloaded_files} файлов")


def get_soup(url):
    time.sleep(DELAY_BETWEEN_REQUESTS)
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            return BeautifulSoup(response.text, 'html.parser')
        else:
            print(f"Ошибка при получении страницы {url}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Ошибка при запросе {url}: {e}")
        return None


def get_subcategories(category_url):
    soup = get_soup(category_url)
    if not soup:
        return []

    subcategories = []
    # Ищем блок "категории" и собираем все ссылки из него
    category_div = soup.find('div', string=lambda t: t and "категории:" in t.lower())
    if category_div and category_div.find_next('ul'):
        category_links = category_div.find_next('ul').find_all('a')
        for link in category_links:
            if 'href' in link.attrs:
                subcategory_url = link['href']
                subcategory_name = link.text.strip()
                subcategories.append({
                    'name': subcategory_name,
                    'url': subcategory_url
                })

    return subcategories


def get_brands(subcategory_url):
    soup = get_soup(subcategory_url)
    if not soup:
        return []

    brands = []
    # Ищем блок "бренды" и собираем все ссылки из него
    brand_div = soup.find('div', string=lambda t: t and "бренды:" in t.lower())
    if brand_div and brand_div.find_next('ul'):
        brand_links = brand_div.find_next('ul').find_all('a')
        for link in brand_links:
            if 'href' in link.attrs:
                brand_url = link['href']
                brand_name = link.text.strip()
                brands.append({
                    'name': brand_name,
                    'url': brand_url
                })

    return brands


def get_models(brand_url):
    soup = get_soup(brand_url)
    if not soup:
        return []

    models = []

    # На странице бренда модели представлены как h2 с ссылками внутри li
    model_items = soup.select('li h2 a')

    # Если не нашли модели через h2, попробуем другие селекторы
    if not model_items:
        model_items = soup.select('li a[href*="/"]')

    for link in model_items:
        model_name = link.text.strip()
        model_url = link['href']

        # Пропускаем навигационные элементы
        nav_terms = ['главная', 'карта сайта', 'контакты', 'о нас', 'блог', 'каталог',
                     'производители', 'устройства', 'меню', 'home', 'sitemap']

        if model_name.lower() not in nav_terms and len(model_name) > 1:
            # Преобразуем относительные URL в абсолютные
            if not model_url.startswith(('http://', 'https://')):
                model_url = urljoin(brand_url, model_url)

            models.append({
                'name': model_name,
                'url': model_url
            })

    # Удаляем дубликаты по URL
    unique_models = []
    seen_urls = set()
    for model in models:
        if model['url'] not in seen_urls:
            seen_urls.add(model['url'])
            unique_models.append(model)

    if not unique_models:
        print(f"DEBUG: Не найдены модели на странице {brand_url}")

    return unique_models


def get_download_url(model_url):
    soup = get_soup(model_url)
    if not soup:
        return None

    # Ищем ссылку для загрузки с классом downloadfilelink
    download_link = soup.select_one('.downloadfilelink')

    if download_link and 'href' in download_link.attrs:
        download_url = download_link['href']
        # Обрабатываем относительные URL
        if not download_url.startswith(('http://', 'https://')):
            download_url = urljoin(model_url, download_url)
        return download_url

    # Если не нашли с классом, ищем любые ссылки на PDF или другие файлы
    pdf_links = soup.select('a[href$=".pdf"], a[href$=".zip"], a[href$=".doc"], a[href$=".docx"]')
    if pdf_links:
        pdf_url = pdf_links[0]['href']
        if not pdf_url.startswith(('http://', 'https://')):
            pdf_url = urljoin(model_url, pdf_url)
        return pdf_url

    return None

def download_file(url, destination):
    global downloaded_files, downloaded_files_list

    if url in downloaded_files_list:
        print(f"Пропуск файла (уже скачан): {url}")
        return True

    time.sleep(DELAY_BETWEEN_REQUESTS)
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=60)

        if response.status_code == 200:
            file_size = int(response.headers.get('content-length', 0))
            block_size = 1024

            with open(destination, 'wb') as file, tqdm(
                    desc=os.path.basename(destination),
                    total=file_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    bar.update(len(data))
                    file.write(data)

            downloaded_files += 1
            downloaded_files_list.append(url)

            if downloaded_files % 10 == 0:
                save_state()

            return True
        else:
            print(f"Ошибка при скачивании файла {url}: {response.status_code}")
            return False
    except Exception as e:
        print(f"Ошибка при скачивании {url}: {e}")
        return False


def extract_zip_if_needed(file_path, output_dir):
    # Если файл - архив ZIP, распаковываем его
    if file_path.lower().endswith('.zip'):
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                pdf_files = [f for f in file_list if f.lower().endswith('.pdf')]

                if pdf_files:
                    for pdf in pdf_files:
                        zip_ref.extract(pdf, output_dir)
                    print(f"Распаковано {len(pdf_files)} PDF-файлов из {file_path}")
                    return True
                else:
                    print(f"В архиве {file_path} не найдены PDF-файлы")
                    return False
        except Exception as e:
            print(f"Ошибка при распаковке {file_path}: {e}")
            return False
    # Если файл не ZIP, ничего делать не нужно
    return True

def process_model(model, brand_name, subcategory_name, category_name):
    global downloaded_files

    model_name = model['name']
    model_url = model['url']

    print(f"Обработка модели: {model_name}")

    # Получаем ссылку для скачивания
    download_url = get_download_url(model_url)

    if not download_url:
        print(f"Не найдена ссылка для скачивания модели {model_name}")
        return False  # Добавлен возврат значения False

    # Определяем имя файла из URL или формируем его из модели
    file_name = os.path.basename(download_url) if download_url.find('/') >-1 else ""
    if not file_name or file_name.strip() == "":
        # Формируем имя файла из модели и бренда
        extension = '.pdf' if download_url.lower().endswith('.pdf') else '.zip'
        file_name = f"{brand_name}_{model_name}{extension}".replace(' ', '_')

    # Очищаем имя файла от недопустимых символов
    file_name = "".join(c for c in file_name if c.isalnum() or c in "._- ")

    # Добавляем информацию о категории и подкатегории в имя файла для сохранения контекста
    file_base, file_ext = os.path.splitext(file_name)
    safe_category = "".join(c for c in category_name if c.isalnum() or c in "._- ")
    safe_subcategory = "".join(c for c in subcategory_name if c.isalnum() or c in "._- ")
    file_name = f"{safe_category}_{safe_subcategory}_{file_base}{file_ext}"

    # Сохраняем все файлы в общую директорию без вложенных папок
    output_path = os.path.join(OUTPUT_DIR, file_name)

    # Проверяем, существует ли файл
    if os.path.exists(output_path):
        print(f"Файл {file_name} уже существует, пропускаем")
        return False  # Добавлен возврат значения False

    # Скачиваем файл
    if download_file(download_url, output_path):
        print(f"Скачан файл {file_name} ({downloaded_files} из {MAX_FILES})")

        # Распаковываем ZIP-файлы, если необходимо
        if output_path.lower().endswith('.zip'):
            extract_zip_if_needed(output_path, OUTPUT_DIR)

        return True  # Добавлен возврат значения True
    else:
        return False  # Добавлен возврат значения False


def main():
    global downloaded_files

    print(f"Начинаем скачивание до {MAX_FILES} инструкций...")

    for main_category in MAIN_CATEGORIES:
        if downloaded_files >= MAX_FILES:
            break

        category_url = main_category['url']
        category_name = main_category['name']
        print(f"\nПолучение подкатегорий для {category_name}...")

        subcategories = get_subcategories(category_url)

        if not subcategories:
            print(f"Подкатегории не найдены для {category_name}. Используем основную категорию.")
            subcategories = [{'name': category_name, 'url': category_url}]

        # Перебираем ВСЕ подкатегории
        for subcategory in subcategories:
            if downloaded_files >= MAX_FILES:
                break

            subcategory_name = subcategory['name']
            subcategory_url = subcategory['url']
            print(f"\nОбработка подкатегории: {subcategory_name}")

            brands = get_brands(subcategory_url)
            print(f"Найдено {len(brands)} брендов в подкатегории {subcategory_name}")

            if not brands:
                continue

            # Ограничиваем количество брендов до MAX_BRANDS_PER_CATEGORY, если их больше
            brands_to_process = brands[:MAX_BRANDS_PER_CATEGORY]

            # Перебираем все выбранные бренды
            for brand in brands_to_process:
                if downloaded_files >= MAX_FILES:
                    break

                brand_name = brand['name']
                brand_url = brand['url']
                print(f"Обработка бренда: {brand_name}")

                models = get_models(brand_url)
                print(f"Найдено {len(models)} моделей у бренда {brand_name}")

                if not models:
                    continue

                # Ограничиваем количество моделей до MAX_MODELS_PER_BRAND
                models_to_process = models[:MAX_MODELS_PER_BRAND]

                # Обрабатываем все выбранные модели
                for model in models_to_process:
                    if downloaded_files >= MAX_FILES:
                        break

                    try:
                        process_model(model, brand_name, subcategory_name, category_name)
                    except Exception as e:
                        print(f"Ошибка при обработке модели {model['name']}: {e}")
                        continue

    print(f"\nГотово! Скачано {downloaded_files} инструкций.")
    save_state()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрерывание пользователем. Сохранение состояния...")
        save_state()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        save_state()
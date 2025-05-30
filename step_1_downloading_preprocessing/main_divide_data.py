import pandas as pd
import numpy as np

# 1. Загрузка данных
df = pd.read_csv("improvement_candidates_5.csv")
texts = df["text"].tolist()

# 2. Перемешивание текстов для случайного распределения
np.random.seed(42)  # для воспроизводимости результатов
np.random.shuffle(texts)

# 3. Расчет количества текстов для каждого файла
total_texts = len(texts)
print(f"Всего текстов: {total_texts}")

# Процентное распределение
distribution = {
    "claude_1": 0.1,
    "claude_2": 0.1,
    "gpt_1": 0.1,
    "gpt_2": 0.1,
    "gpt_3": 0.1,
    "deepseek": 0.1,
    "llama_1": 0.1,
    "llama_2": 0.1,
    "gemini_1": 0.1,
    "gemini_2": 0.1
}

# Проверка на сумму процентов = 100% с небольшим допуском для чисел с плавающей точкой
sum_distribution = sum(distribution.values())
print(f"Сумма процентов: {sum_distribution}")
assert abs(sum_distribution - 1.0) < 1e-10, f"Сумма процентов должна быть равна 1.0, получено {sum_distribution}"

# 4. Определение точного количества текстов для каждой модели
# Создаем словарь для хранения количества текстов для каждой модели
counts = {}
texts_left = total_texts
models_left = list(distribution.keys())

# Распределяем тексты между моделями
for model in models_left[:-1]:  # Обрабатываем все модели, кроме последней
    percentage = distribution[model]
    count = int(total_texts * percentage)
    counts[model] = count
    texts_left -= count

# Последней модели отдаем все оставшиеся тексты
counts[models_left[-1]] = texts_left

# Вывод информации о распределении
print("\nРаспределение текстов:")
for model, count in counts.items():
    percentage = count / total_texts * 100
    print(f"{model}: {count} текстов ({percentage:.2f}%)")

print(f"Всего распределено: {sum(counts.values())} текстов из {total_texts}")

# 5. Распределение и сохранение текстов
start_idx = 0
for model_name, count in counts.items():
    # Получение среза текстов
    end_idx = start_idx + count
    model_texts = texts[start_idx:end_idx]

    # Создание и сохранение DataFrame
    model_df = pd.DataFrame({"text": model_texts})
    filename = f"{model_name}.csv"
    model_df.to_csv(filename, index=False)

    percentage = count / total_texts * 100
    print(f"Сохранено {len(model_texts)} текстов в файл {filename} ({percentage:.2f}%)")

    # Обновление начального индекса для следующей итерации
    start_idx = end_idx

print(f"\nУспешно распределено: {start_idx} текстов из {total_texts}")


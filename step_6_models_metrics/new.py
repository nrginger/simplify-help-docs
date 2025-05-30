import pandas as pd

# 1. Загрузка полного набора и уже отобранного
full_df   = pd.read_csv('combined_results.csv')
prev_df   = pd.read_csv('combined_sample50.csv')

# 2. Исключаем тексты, которые уже были в первой выборке
mask      = ~full_df['original_text'].isin(prev_df['original_text'])
remaining = full_df[mask]

# 3. Берём 50 новых случайных строк
new_sample = remaining.sample(n=50, random_state=123).reset_index(drop=True)

# 4. Сохраняем их в файл
new_sample.to_csv('combined_sample50_round2.csv', index=False)

print("Новая выборка из 50 текстов сохранена в combined_sample50_round2.csv")
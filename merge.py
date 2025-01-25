import pandas as pd

def merge_rsu_files(file1, file2, output):
    # Читаем файлы
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Получаем все уникальные столбцы из обоих файлов
    all_columns = list(set(df1.columns) | set(df2.columns))
    
    # Добавляем отсутствующие столбцы в каждый датафрейм с пустыми значениями
    for col in all_columns:
        if col not in df1.columns:
            df1[col] = ""
        if col not in df2.columns:
            df2[col] = ""
    
    # Объединяем датафреймы
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Сортируем по дате
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date', ascending=False)
    
    # Сохраняем результат
    df.to_csv(output, index=False, na_rep='')

# Использование:
merge_rsu_files('EquityAwardsCenter_Transactions_20250125150857.csv', 
                'EquityAwardsCenter_Transactions_20250125150834.csv',
                'merged_equity.csv')
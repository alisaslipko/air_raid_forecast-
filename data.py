import pandas as pd

def load_and_inspect_missing_data(url="https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/refs/heads/main/datasets/official_data_en.csv"):
    print("--- 1. Завантаження та Пропуски ---")
    df = pd.read_csv(url, parse_dates=['started_at', 'finished_at'])
    df_2026 = df[df['started_at'].dt.year == 2026]
    
    slices = {"Весь період": df, "2026 рік": df_2026}
    
    for name, data in slices.items():
        print(f"\n[{name}]")
        print(f"Кількість записів: {len(data)}")
        
        # Аналіз пропусків у часових мітках
        missing_finished = data['finished_at'].isna().sum()
        missing_pct = (missing_finished / len(data)) * 100 if len(data) > 0 else 0
        print(f"Незавершені тривоги (NaN у 'finished_at'): {missing_finished} ({missing_pct:.2f}%)")
        
        # Перевірка наявності інших пропусків (область)
        missing_oblast = data['oblast'].isna().sum()
        if missing_oblast > 0:
            print(f"Пропуски в 'oblast': {missing_oblast}")
    return df

def merge_oblast_intervals(df):
    """
    Об'єднує (зливає) пересічні інтервали тривог на рівні області.
    """
    print("--- 1. Злиття інтервалів на рівні областей ---")
    
    # Відкидаємо записи без області або без часу завершення
    df_clean = df.dropna(subset=['oblast', 'started_at', 'finished_at']).copy()
    
    # Сортуємо по області та часу початку для коректного алгоритму злиття
    df_clean = df_clean.sort_values(['oblast', 'started_at'])
    
    merged_records = []
    
    # Проходимо по кожній області окремо
    for oblast, group in df_clean.groupby('oblast'):
        current_start = None
        current_end = None
        
        for _, row in group.iterrows():
            start = row['started_at']
            end = row['finished_at']
            
            # Ініціалізація першого інтервалу
            if current_start is None:
                current_start = start
                current_end = end
            # Якщо поточна тривога починається до того, як закінчилася попередня -> злиття
            elif start <= current_end:
                current_end = max(current_end, end)
            # Якщо інтервали не перетинаються -> зберігаємо поточний і починаємо новий
            else:
                merged_records.append({
                    'oblast': oblast, 
                    'started_at': current_start, 
                    'finished_at': current_end
                })
                current_start = start
                current_end = end
                
        # Зберігаємо останній інтервал для області
        if current_start is not None:
            merged_records.append({
                'oblast': oblast, 
                'started_at': current_start, 
                'finished_at': current_end
            })
            
    df_merged = pd.DataFrame(merged_records)
    print(f"Кількість сирих записів: {len(df)}")
    print(f"Кількість унікальних епізодів після злиття: {len(df_merged)}")
    
    return df_merged

def clean_timezones(df_intervals: pd.DataFrame) -> pd.DataFrame:
    """
    Приводить стовпці з часом до локального часу (Europe/Kyiv) 
    та видаляє метадані таймзони (tz-naive), щоб уникнути конфліктів з NumPy.
    """
    print("--- Очищення часових зон (перехід до tz-naive) ---")
    df_out = df_intervals.copy()
    
    for col in ['started_at', 'finished_at']:
        # Перевіряємо, чи є метадані про таймзону
        if df_out[col].dt.tz is not None:
            # Переводимо в Київський час і відкидаємо таймзону
            df_out[col] = df_out[col].dt.tz_convert('Europe/Kyiv').dt.tz_localize(None)
            
    return df_out

def crop_raw_intervals(df_intervals: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """
    Обрізає сирі дані до вказаної дати (наприклад, '2024-07-01').
    Правильно обробляє перехідні тривоги, що почалися до start_date, а закінчилися після.
    """
    print(f"--- Обрізка сирих даних з {start_date} ---")
    df_out = df_intervals.copy()
    start_dt = pd.to_datetime(start_date)
    
    # 1. Залишаємо тільки ті тривоги, які закінчилися ПІСЛЯ або ТОЧНО в дату старту
    df_out = df_out[df_out['finished_at'] >= start_dt].copy()
    
    # 2. Якщо тривога почалася до дати старту (перехідна), штучно підтягуємо її початок до start_date
    df_out['started_at'] = df_out['started_at'].clip(lower=start_dt)
    
    # Сортуємо для коректної роботи merge_asof у майбутньому
    df_out = df_out.sort_values(['oblast', 'started_at']).reset_index(drop=True)
    
    print(f"Залишилось рядків (інтервалів): {len(df_out)}")
    return df_out

def remove_micro_alerts(df, min_minutes=2):
    """
    Видаляє тривоги, тривалість яких менша за вказаний поріг.
    """
    print(f"--- Видалення мікротривог (< {min_minutes} хв) ---")
    
    initial_count = len(df)
    
    # Обчислюємо тривалість у хвилинах
    duration_mins = (df['finished_at'] - df['started_at']).dt.total_seconds() / 60.0
    
    # Фільтруємо
    df_filtered = df[duration_mins >= min_minutes].copy()
    
    removed_count = initial_count - len(df_filtered)
    
    print(f"Вилучено мікротривог: {removed_count}")
    print(f"Залишилося записів: {len(df_filtered)}")
    
    return df_filtered

def remove_static_regions(df):
    """
    Видаляє регіони з постійною тривогою (константи) з датасету.
    """
    print("--- Видалення статичних регіонів ---")
    
    regions_to_remove = [
        'Luhanska oblast', 
    ]
    
    initial_count = len(df)
    
    # Фільтрація: залишаємо лише ті записи, де 'oblast' не входить до списку на видалення
    df_filtered = df[~df['oblast'].isin(regions_to_remove)].copy()
    
    removed_count = initial_count - len(df_filtered)
    
    print(f"Видалені регіони: {', '.join(regions_to_remove)}")
    print(f"Вилучено записів: {removed_count}")
    print(f"Залишилося записів для аналізу: {len(df_filtered)}")
    
    return df_filtered
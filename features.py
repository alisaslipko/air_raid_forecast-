from data import *
import pandas as pd
import numpy as np

def build_base_panel(df_intervals, freq='15min'):
    print("--- Побудова базової панелі (Точковий стан + інтервальний таргет) ---")
    
    # Визначаємо глобальні межі
    min_time = df_intervals['started_at'].min().floor('h')
    max_time = df_intervals['finished_at'].max().ceil('h')
    time_grid = pd.date_range(start=min_time, end=max_time, freq=freq)
    
    dfs = []

    global_end = df_intervals['finished_at'].max().to_datetime64()
    blind_spot_start = global_end - np.timedelta64(120, 'm')
    
    for oblast, group in df_intervals.groupby('oblast'):
        # 1. Ініціалізація сітки для області
        df_grid = pd.DataFrame({'timestamp': time_grid, 'oblast': oblast})
        
        # 2. Поточний стан (S_{i,T}) та current_duration_mins
        # Шукаємо останній інтервал, який ПОЧАВСЯ до або рівно в момент T
        group_starts = group[['started_at', 'finished_at']].sort_values('started_at')
        merged_starts = pd.merge_asof(
            df_grid, group_starts,
            left_on='timestamp', right_on='started_at',
            direction='backward'
        )
        
        # Тривога активна в момент T, якщо знайдений інтервал ще не завершився
        df_grid['is_alert'] = (merged_starts['timestamp'] < merged_starts['finished_at']).astype(int)
        
        # Розрахунок тривалості поточної тривоги (якщо не активна — 0)
        df_grid['current_duration_mins'] = 0.0
        active_mask = df_grid['is_alert'] == 1
        df_grid.loc[active_mask, 'current_duration_mins'] = (
            df_grid.loc[active_mask, 'timestamp'] - merged_starts.loc[active_mask, 'started_at']
        ).dt.total_seconds() / 60.0
        
        # 3. time_since_last_alert
        # Шукаємо останній інтервал, який ЗАВЕРШИВСЯ до або рівно в момент T
        group_ends = group[['started_at', 'finished_at']].sort_values('finished_at')
        merged_ends = pd.merge_asof(
            df_grid, group_ends,
            left_on='timestamp', right_on='finished_at',
            direction='backward'
        )
        
        # Розраховуємо лише для моментів без тривоги
        df_grid['time_since_last_alert'] = np.nan
        inactive_mask = df_grid['is_alert'] == 0
        df_grid.loc[inactive_mask, 'time_since_last_alert'] = (
            df_grid.loc[inactive_mask, 'timestamp'] - merged_ends.loc[inactive_mask, 'finished_at']
        ).dt.total_seconds() / 60.0
        
        # Якщо тривога активна, час з останньої = 0
        df_grid.loc[active_mask, 'time_since_last_alert'] = 0.0
        
        # 4. Цільова змінна Y_{i,T}: Тривога в (T, T + 60]
        # Математична умова: start <= T + 60 AND end > T
        # ОНОВЛЕНИЙ ТАРГЕТ Y_{i,T}: (T + 60, T + 120]
        T_arr = df_grid['timestamp'].values
        Y_arr = np.zeros(len(T_arr), dtype=float)
        starts_arr = group['started_at'].values
        ends_arr = group['finished_at'].values
        
        for s, e in zip(starts_arr, ends_arr):
            # Формула перетину: T > s - 120 хвилин ТА T < e - 60 хвилин
            s_minus_120 = s - np.timedelta64(120, 'm')
            e_minus_60 = e - np.timedelta64(60, 'm')
            
            mask = (T_arr > s_minus_120) & (T_arr < e_minus_60)
            Y_arr[mask] = 1.0
            
        df_grid['target_60_120m'] = Y_arr
        
        df_grid.loc[T_arr > blind_spot_start, 'target_60_120m'] = np.nan

        dfs.append(df_grid)
        
    df_panel = pd.concat(dfs, ignore_index=True)
    
    # 5. Видалення невідомих ділянок (початок датасету та сліпа зона)
    df_panel = df_panel.dropna(subset=['time_since_last_alert', 'target_60_120m']).copy()
    
    df_panel = df_panel.sort_values(['oblast', 'timestamp']).reset_index(drop=True)
    
    # 6. Лаги стану 
    for lag in range(1, 5):
        df_panel[f'lag_{lag}'] = df_panel.groupby('oblast')['is_alert'].shift(lag).fillna(0)
    
    # Типізація
    df_panel['oblast'] = df_panel['oblast'].astype('category')
    df_panel['target_60_120m'] = df_panel['target_60_120m'].astype(int)
    
    print(f"Розмір панелі після обрізки: {df_panel.shape}")
    return df_panel

def add_history_features(df_panel, df_intervals):
    print("--- Розрахунок ознак режиму (alert_pct та медіани) ---")
    df_int = df_intervals.copy()
    df_int['dur_mins'] = (df_int['finished_at'] - df_int['started_at']).dt.total_seconds() / 60.0
    df_int = df_int.sort_values(['oblast', 'started_at'])
    df_int['cum_dur_before'] = df_int.groupby('oblast')['dur_mins'].cumsum() - df_int['dur_mins']
    
    dfs_out = []
    windows = {'24h': 24*60, '7d': 7*24*60, '30d': 30*24*60}
    
    for oblast, group_panel in df_panel.groupby('oblast'):
        group_int = df_int[df_int['oblast'] == oblast].copy()
        if group_int.empty:
            for w_name in windows.keys(): group_panel[f'alert_pct_{w_name}'] = 0.0
            group_panel['median_episode_dur_7d'] = 0.0
            dfs_out.append(group_panel)
            continue
            
        T_series = group_panel['timestamp']
        T_arr = T_series.values
        
        def get_C_T(T_targets):
            df_T = pd.DataFrame({'T': T_targets})
            merged = pd.merge_asof(
                df_T, group_int[['started_at', 'finished_at', 'cum_dur_before']],
                left_on='T', right_on='started_at',
                direction='backward'
            )
            merged['cum_dur_before'] = merged['cum_dur_before'].fillna(0)
            merged['started_at'] = merged['started_at'].fillna(merged['T']) 
            merged['finished_at'] = merged['finished_at'].fillna(merged['T'])
            
            current_overlap = (merged[['T', 'finished_at']].min(axis=1) - merged['started_at']).dt.total_seconds() / 60.0
            current_overlap = current_overlap.clip(lower=0)
            
            # ВИПРАВЛЕННЯ: Повертаємо чистий масив NumPy, щоб уникнути конфлікту індексів Pandas
            return (merged['cum_dur_before'] + current_overlap).values

        C_T = get_C_T(T_series)
        for w_name, w_mins in windows.items():
            T_minus_W = T_series - pd.Timedelta(minutes=w_mins)
            C_T_minus_W = get_C_T(T_minus_W)
            
            # ВИПРАВЛЕННЯ: Векторизований розрахунок з NumPy-масивами
            pct = (C_T - C_T_minus_W) / w_mins
            group_panel[f'alert_pct_{w_name}'] = np.clip(np.nan_to_num(pct, nan=0.0), 0, 1)
            
        group_ends = group_int.sort_values('finished_at')
        ends_arr = group_ends['finished_at'].values
        durs_arr = group_ends['dur_mins'].values
        
        T_minus_7d_arr = T_arr - np.timedelta64(7, 'D')
        left_idxs = np.searchsorted(ends_arr, T_minus_7d_arr, side='right')
        right_idxs = np.searchsorted(ends_arr, T_arr, side='right')
        
        medians = np.zeros(len(T_arr), dtype=float)
        for i in range(len(T_arr)):
            l, r = left_idxs[i], right_idxs[i]
            if l < r: medians[i] = np.median(durs_arr[l:r])
                
        group_panel['median_episode_dur_7d'] = medians
        dfs_out.append(group_panel)
        
    return pd.concat(dfs_out, ignore_index=True).sort_values(['oblast', 'timestamp']).reset_index(drop=True)

def add_spatial_features(df_panel):
    print("--- Розрахунок просторових ознак ---")
    
    # Граф сусідства для 24 регіонів з датасету
    neighbors_map = {
        'Cherkaska oblast': ['Kyivska oblast', 'Poltavska oblast', 'Kirovohradska oblast', 'Vinnytska oblast'],
        'Chernihivska oblast': ['Kyivska oblast', 'Sumska oblast', 'Poltavska oblast'],
        'Chernivetska oblast': ['Ivano-Frankivska oblast', 'Ternopilska oblast', 'Khmelnytska oblast', 'Vinnytska oblast'],
        'Dnipropetrovska oblast': ['Poltavska oblast', 'Kharkivska oblast', 'Donetska oblast', 'Zaporizka oblast', 'Khersonska oblast', 'Mykolaivska oblast', 'Kirovohradska oblast'],
        'Donetska oblast': ['Kharkivska oblast', 'Dnipropetrovska oblast', 'Zaporizka oblast'],
        'Ivano-Frankivska oblast': ['Zakarpatska oblast', 'Lvivska oblast', 'Ternopilska oblast', 'Chernivetska oblast'],
        'Kharkivska oblast': ['Sumska oblast', 'Poltavska oblast', 'Dnipropetrovska oblast', 'Donetska oblast'],
        'Khersonska oblast': ['Mykolaivska oblast', 'Dnipropetrovska oblast', 'Zaporizka oblast'],
        'Khmelnytska oblast': ['Rivnenska oblast', 'Zhytomyrska oblast', 'Vinnytska oblast', 'Chernivetska oblast', 'Ternopilska oblast'],
        'Kirovohradska oblast': ['Cherkaska oblast', 'Poltavska oblast', 'Dnipropetrovska oblast', 'Mykolaivska oblast', 'Odeska oblast', 'Vinnytska oblast'],
        'Kyiv City': ['Kyivska oblast'],
        'Kyivska oblast': ['Kyiv City', 'Zhytomyrska oblast', 'Vinnytska oblast', 'Cherkaska oblast', 'Poltavska oblast', 'Chernihivska oblast'],
        'Lvivska oblast': ['Volynska oblast', 'Rivnenska oblast', 'Ternopilska oblast', 'Ivano-Frankivska oblast', 'Zakarpatska oblast'],
        'Mykolaivska oblast': ['Odeska oblast', 'Kirovohradska oblast', 'Dnipropetrovska oblast', 'Khersonska oblast'],
        'Odeska oblast': ['Vinnytska oblast', 'Kirovohradska oblast', 'Mykolaivska oblast'],
        'Poltavska oblast': ['Chernihivska oblast', 'Sumska oblast', 'Kharkivska oblast', 'Dnipropetrovska oblast', 'Kirovohradska oblast', 'Cherkaska oblast', 'Kyivska oblast'],
        'Rivnenska oblast': ['Volynska oblast', 'Zhytomyrska oblast', 'Khmelnytska oblast', 'Ternopilska oblast', 'Lvivska oblast'],
        'Sumska oblast': ['Chernihivska oblast', 'Poltavska oblast', 'Kharkivska oblast'],
        'Ternopilska oblast': ['Rivnenska oblast', 'Khmelnytska oblast', 'Chernivetska oblast', 'Ivano-Frankivska oblast', 'Lvivska oblast'],
        'Vinnytska oblast': ['Zhytomyrska oblast', 'Kyivska oblast', 'Cherkaska oblast', 'Kirovohradska oblast', 'Odeska oblast', 'Chernivetska oblast', 'Khmelnytska oblast'],
        'Volynska oblast': ['Rivnenska oblast', 'Lvivska oblast'],
        'Zakarpatska oblast': ['Lvivska oblast', 'Ivano-Frankivska oblast'],
        'Zaporizka oblast': ['Khersonska oblast', 'Dnipropetrovska oblast', 'Donetska oblast'],
        'Zhytomyrska oblast': ['Rivnenska oblast', 'Khmelnytska oblast', 'Vinnytska oblast', 'Kyivska oblast']
    }
    
    # 1. Визначаємо, чи тривога "нова" (dS_k,T)
    # Згідно з формалізацією: max(0, S_T - S_{T - dt}). 
    # Використовуємо існуючий lag_1.
    df_panel['is_new_alert'] = ((df_panel['is_alert'] == 1) & (df_panel['lag_1'] == 0)).astype(int)
    
    # 2. Розгортання у широкі матриці для векторизації (Рядки: час, Колонки: області)
    pivot_alert = df_panel.pivot(index='timestamp', columns='oblast', values='is_alert')
    pivot_new = df_panel.pivot(index='timestamp', columns='oblast', values='is_new_alert')
    
    total_regions = len(pivot_alert.columns)
    
    # Загальнонаціональні суми (Скільки всього областей активні зараз)
    total_active_ua = pivot_alert.sum(axis=1)
    total_new_ua = pivot_new.sum(axis=1)
    
    spatial_features = []
    
    for oblast in pivot_alert.columns:
        oblast_alert = pivot_alert[oblast]
        oblast_new = pivot_new[oblast]
        
        # --- Глобальні ознаки (виключаємо саму цільову область i) ---
        ua_act_pct = (total_active_ua - oblast_alert) / (total_regions - 1)
        ua_new_pct = (total_new_ua - oblast_new) / (total_regions - 1)
        
        # --- Локальні ознаки (сусіди) ---
        neighbors = neighbors_map.get(oblast, [])
        valid_neighbors = [n for n in neighbors if n in pivot_alert.columns]
        n_count = len(valid_neighbors)
        
        if n_count > 0:
            neigh_act_pct = pivot_alert[valid_neighbors].sum(axis=1) / n_count
            neigh_new_pct = pivot_new[valid_neighbors].sum(axis=1) / n_count
        else:
            # Технічний запобіжник
            neigh_act_pct = pd.Series(0.0, index=pivot_alert.index)
            neigh_new_pct = pd.Series(0.0, index=pivot_alert.index)
            
        # Пакуємо результати для поточної області
        df_obl = pd.DataFrame({
            'timestamp': pivot_alert.index,
            'oblast': oblast,
            'neighbors_active_pct': neigh_act_pct.values,
            'neighbors_new_alerts_pct': neigh_new_pct.values,
            'ua_active_pct': ua_act_pct.values,
            'ua_new_alerts_pct': ua_new_pct.values
        })
        spatial_features.append(df_obl)
        
    # 3. Об'єднання та приєднання до вихідної панелі
    df_spatial = pd.concat(spatial_features, ignore_index=True)
    df_panel = df_panel.merge(df_spatial, on=['timestamp', 'oblast'], how='left')
    
    # Видаляємо тимчасову колонку
    df_panel = df_panel.drop(columns=['is_new_alert'])
    
    return df_panel

def add_calendar_features(df_panel):
    print("--- Розрахунок календарних ознак (Гармоніки) ---")
    
    timestamps = df_panel['timestamp']
    
    # 1. Час доби
    fractional_hour = timestamps.dt.hour + timestamps.dt.minute / 60.0
    df_panel['hour_sin'] = np.sin(2 * np.pi * fractional_hour / 24.0)
    df_panel['hour_cos'] = np.cos(2 * np.pi * fractional_hour / 24.0)
    
    # 3. День тижня (0-6)
    day_of_week = timestamps.dt.dayofweek
    df_panel['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
    df_panel['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)
        
    return df_panel

def generate_downsampled_grids(df_15m: pd.DataFrame) -> dict:
    """
    Створює 30-хвилинну та 60-хвилинну сітки шляхом чистого зрізу з 15-хвилинної.
    Ніякі ознаки не перераховуються.
    """
    print("--- Генерація даунсемпл-сіток ---")
    
    # Фільтруємо за хвилинами
    df_30m = df_15m[df_15m['timestamp'].dt.minute.isin([0, 30])].reset_index(drop=True)
    df_60m = df_15m[df_15m['timestamp'].dt.minute == 0].reset_index(drop=True)
    
    grids = {
        '15m': df_15m,
        '30m': df_30m,
        '60m': df_60m
    }
    
    print(f"15 хв: {df_15m.shape[0]} рядків")
    print(f"30 хв: {df_30m.shape[0]} рядків")
    print(f"60 хв: {df_60m.shape[0]} рядків")
    
    return grids

def analyze_class_imbalance(df: pd.DataFrame, grid_name: str) -> None:
    """
    Аналізує та виводить статистику цільової змінної глобально та 
    в розрізі поточного стану (є тривога / немає тривоги).
    """
    target_col = 'target_60_120m'
    
    print(f"\n=== Аналіз дисбалансу: Сітка {grid_name} ===")
    
    # Глобальний
    glob_counts = df[target_col].value_counts(normalize=True) * 100
    print(f"Глобально:")
    print(f"  0 (Спокій):  {glob_counts.get(0, 0):.2f}%")
    print(f"  1 (Тривога): {glob_counts.get(1, 0):.2f}%")
    
    # Стан: Зараз МИР (is_alert == 0)
    df_peace = df[df['is_alert'] == 0]
    peace_counts = df_peace[target_col].value_counts(normalize=True) * 100
    print(f"\nЯкщо зараз НЕМАЄ тривоги (is_alert=0):")
    print(f"  Залишиться спокій (0): {peace_counts.get(0, 0):.2f}%")
    print(f"  Почнеться тривога (1): {peace_counts.get(1, 0):.2f}%")
    
    # Стан: Зараз ТРИВОГА (is_alert == 1)
    df_alert = df[df['is_alert'] == 1]
    alert_counts = df_alert[target_col].value_counts(normalize=True) * 100
    print(f"\nЯкщо зараз Є тривога (is_alert=1):")
    print(f"  Закінчиться (0):   {alert_counts.get(0, 0):.2f}%")
    print(f"  Продовжиться (1):  {alert_counts.get(1, 0):.2f}%")
    print("-" * 40)

def build_feature_panel(df_merged_oblasts, freq='15min', start_date='2024-07-01'):

    df_merged_oblasts = clean_timezones(df_merged_oblasts)
    df_merged_oblasts = crop_raw_intervals(df_merged_oblasts, start_date)

    # 2. Генерація базової панелі
    df_panel = build_base_panel(df_merged_oblasts, freq=freq)

    # 3. Додавання історичних ознак режиму
    df_panel = add_history_features(df_panel, df_merged_oblasts)

    # 4. Додавання просторових ознак
    df_panel = add_spatial_features(df_panel)

    # 5. Додавання календарних ознак
    df_panel = add_calendar_features(df_panel)

    return df_panel

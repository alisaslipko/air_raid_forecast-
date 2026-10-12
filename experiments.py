from models import *
from evaluation import *
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

def run_experiment(df_grid: pd.DataFrame, 
                   grid_name: str, 
                   lgbm_params: dict = None):
    """
    Повний цикл: розбиття, навчання 3-х моделей, чесна оцінка на валідації 
    та візуалізація метрик.
    """
    print(f"\n{'='*50}")
    print(f"🚀 ЗАПУСК ЕКСПЕРИМЕНТУ: Сітка {grid_name}")
    print(f"{'='*50}")
    
    # 1. Розбиття датасету
    df_train, df_val, df_test = time_series_split(
        df=df_grid, 
        val_months=2, test_months=2, gap_hours=2, burn_in_days=30
    )
    
    target_col = 'target_60_120m'
    y_train = df_train[target_col]
    y_val = df_val[target_col]
    
    # 2. Ініціалізація моделей
    if lgbm_params is None:
        lgbm_params = {'n_estimators': 300, 'learning_rate': 0.05}
        
    models = {
        "Naive Baseline": NaiveBaseline(),
        "Tree (1-step)": TreeBaseline(max_depth=5),
        "Main LGBM": MainLGBM(params=lgbm_params, early_stopping_rounds=50)
    }
    
    # 3. Навчання
    print("\n--- НАВЧАННЯ ---")
    for name, model in models.items():
        if name == "Main LGBM":
            model.fit(df_train, y_train, df_val, y_val)
        elif name == "Naive Baseline":
            model.fit(df_train)
        else:
            model.fit(df_train, y_train)
            
    # 4. Оцінка на Валідації (Apples-to-Apples)
    print("\n--- ОЦІНКА НА ВАЛІДАЦІЇ (Common Nodes: хвилина 00) ---")
    
    mask_common_val = (df_val['timestamp'].dt.minute == 0)
    df_val_common = df_val[mask_common_val].copy()
    y_val_common = y_val[mask_common_val].copy()
    is_alert_common = df_val_common['is_alert'].copy()
    
    print(f"Розмір валідації для оцінки (тільки xx:00): {len(df_val_common)} рядків")
    
    results = {}
    predictions = {}
    for name, model in models.items():
        print(f"\n{'='*40}")
        print(f"📊 Модель: {name}")
        
        y_prob = model.predict_proba(df_val_common)[:, 1]
        predictions[name] = y_prob
        
        timings = model.get_timings()
        print(f"⏳ Час навчання: {timings['train_time_sec']:.2f} сек")
        print(f"⏱ Час інференсу (на {len(df_val_common)} рядках): {timings['inference_time_sec']:.4f} сек")
        
        # === НОВЕ: Log Loss для 3-х станів ===
        print("\n📉 Log Loss:")
        masks_eval = {
            "Global": np.ones_like(y_val_common, dtype=bool),
            "Peace (is_alert=0)": (is_alert_common == 0),
            "Alert (is_alert=1)": (is_alert_common == 1)
        }
        for state_name, mask in masks_eval.items():
            if mask.sum() > 0:
                ll = log_loss(y_val_common[mask], y_prob[mask], labels=[0, 1])
                print(f"   - {state_name}: {ll:.4f}")
            else:
                print(f"   - {state_name}: Немає даних")
        
        # Якщо це головна модель — показуємо Топ-10 фічей
        if name == "Main LGBM":
            print("\nТоп-10 найважливіших ознак (Gain):")
            display(model.get_feature_importance().head(10))
            
        # Будуємо графіки
        res_data = plot_threshold_search(y_val_common, y_prob, is_alert_common)
        results[name] = res_data

    print("\n" + "="*80)
    print("ПОРІВНЯЛЬНИЙ ЗВІТ (Threshold = 0.5)")
    print("="*80)
    display_side_by_side_reports(y_val_common, predictions, is_alert_common, threshold=0.5)
        
    return models, results
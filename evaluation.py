import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve, 
                             auc, average_precision_score, precision_score, 
                             recall_score, f1_score, log_loss)
from sklearn.metrics import confusion_matrix, classification_report
from IPython.display import display, HTML

def time_series_split(df: pd.DataFrame, 
                      val_months: int = 2, 
                      test_months: int = 2, 
                      gap_hours: int = 2, 
                      burn_in_days: int = 30):
    """
    Розбиває панель на Train, Validation та Test із заданими "мертвими зонами" 
    та періодом "прогріву" (burn-in).
    """
    print("--- Time-Series Split ---")
    
    # Визначаємо межі доступних даних
    max_time = df['timestamp'].max()
    min_time = df['timestamp'].min()
    
    # Розраховуємо часові мітки розбиття з кінця до початку
    test_start = max_time - pd.DateOffset(months=test_months)
    val_end = test_start - pd.Timedelta(hours=gap_hours)
    
    val_start = val_end - pd.DateOffset(months=val_months)
    train_end = val_start - pd.Timedelta(hours=gap_hours)
    
    # Тренування починається після відкидання burn-in періоду
    train_start = min_time + pd.Timedelta(days=burn_in_days)
    
    print(f"Повний період: {min_time}  -->  {max_time}")
    print(f"Train:         {train_start}  -->  {train_end}")
    print(f"Gap 1:         {gap_hours} години")
    print(f"Validation:    {val_start}  -->  {val_end}")
    print(f"Gap 2:         {gap_hours} години")
    print(f"Test:          {test_start}  -->  {max_time}")
    
    # Фільтруємо датафрейми
    df_train = df[(df['timestamp'] >= train_start) & (df['timestamp'] <= train_end)].copy()
    df_val   = df[(df['timestamp'] >= val_start) & (df['timestamp'] <= val_end)].copy()
    df_test  = df[(df['timestamp'] >= test_start) & (df['timestamp'] <= max_time)].copy()
    
    print(f"\nРозміри вибірок:")
    print(f"Train: {len(df_train)} рядків")
    print(f"Val:   {len(df_val)} рядків")
    print(f"Test:  {len(df_test)} рядків")

    for col in df_train.select_dtypes(include=['object']).columns:
        df_train[col] = df_train[col].astype('category')

    for col in df_val.select_dtypes(include=['object']).columns:
        df_val[col] = df_val[col].astype('category')
    
    return df_train, df_val, df_test

def plot_threshold_search(y_true, y_prob, is_alert_state):
    """
    Будує дашборд із 5 графіків (ROC, PR зверху та Precision, Recall, F1 знизу в ряд).
    Повертає словник із числовими значеннями для кожного стану.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    is_alert_state = np.asarray(is_alert_state)
    
    masks = {
        "Global": np.ones_like(y_true, dtype=bool),
        "Peace (is_alert=0)": (is_alert_state == 0),
        "Alert (is_alert=1)": (is_alert_state == 1)
    }
    colors = {"Global": "black", "Peace (is_alert=0)": "blue", "Alert (is_alert=1)": "red"}
    line_styles = {"Global": "-", "Peace (is_alert=0)": "--", "Alert (is_alert=1)": ":"}
    
    thresholds = np.linspace(0.05, 0.95, 19)
    results_data = {}
    
    fig = plt.figure(figsize=(14, 6.5))
    gs = gridspec.GridSpec(2, 6, figure=fig)
    
    ax_roc = fig.add_subplot(gs[0, 1:3])
    ax_pr  = fig.add_subplot(gs[0, 3:5])
    
    ax_prec = fig.add_subplot(gs[1, 0:2])
    ax_rec  = fig.add_subplot(gs[1, 2:4])
    ax_f1   = fig.add_subplot(gs[1, 4:6])
    
    for state, mask in masks.items():
        if mask.sum() == 0 or len(np.unique(y_true[mask])) < 2:
            continue
            
        y_t = y_true[mask]
        y_p = y_prob[mask]
        
        # --- 1. ROC Curve ---
        fpr, tpr, _ = roc_curve(y_t, y_p)
        roc_auc = roc_auc_score(y_t, y_p)
        ax_roc.plot(fpr, tpr, color=colors[state], linestyle=line_styles[state], 
                    label=f"{state} (AUC={roc_auc:.3f})")
        
        # --- 2. PR Curve (Тільки Average Precision в легенді) ---
        prec_curve, rec_curve, _ = precision_recall_curve(y_t, y_p)
        pr_auc = auc(rec_curve, prec_curve) # Залишаємо розрахунок для словника
        ap = average_precision_score(y_t, y_p)
        
        ax_pr.plot(rec_curve, prec_curve, color=colors[state], linestyle=line_styles[state], 
                   label=f"{state} (AP={ap:.3f})")
        
        # --- 3. Threshold Metrics ---
        prec_list, rec_list, f1_list = [], [], []
        for th in thresholds:
            y_pred = (y_p >= th).astype(int)
            prec_list.append(precision_score(y_t, y_pred, zero_division=0))
            rec_list.append(recall_score(y_t, y_pred, zero_division=0))
            f1_list.append(f1_score(y_t, y_pred, zero_division=0))
            
        ax_prec.plot(thresholds, prec_list, color=colors[state], linestyle=line_styles[state], label=state)
        ax_rec.plot(thresholds, rec_list, color=colors[state], linestyle=line_styles[state], label=state)
        ax_f1.plot(thresholds, f1_list, color=colors[state], linestyle=line_styles[state], label=state)
        
        # --- 4. Збереження результатів ---
        df_thresholds = pd.DataFrame({
            'threshold': thresholds,
            'precision': prec_list,
            'recall': rec_list,
            'f1': f1_list
        })
        
        results_data[state] = {
            'roc': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc},
            'pr_curve': {'precision': prec_curve, 'recall': rec_curve, 'auc': pr_auc, 'ap': ap},
            'threshold_metrics': df_thresholds
        }

    # --- 5. Форматування графіків ---
    for ax in [ax_roc, ax_pr, ax_prec, ax_rec, ax_f1]:
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    ax_roc.set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax_pr.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    
    ax_prec.set(title="Threshold vs Precision", xlabel="Threshold", ylabel="Precision", xticks=np.arange(0.1, 1.0, 0.2))
    ax_rec.set(title="Threshold vs Recall", xlabel="Threshold", ylabel="Recall", xticks=np.arange(0.1, 1.0, 0.2))
    ax_f1.set(title="Threshold vs F1-Score", xlabel="Threshold", ylabel="F1-Score", xticks=np.arange(0.1, 1.0, 0.2))

    plt.tight_layout()
    plt.show()
    
    return results_data


def print_standard_metrics(y_true, y_prob, is_alert_state, threshold=0.5, model_name=""):
    """
    Виводить стандартні confusion_matrix та classification_report 
    для трьох станів (Global, Peace, Alert).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    is_alert_state = np.asarray(is_alert_state)
    y_pred = (y_prob >= threshold).astype(int)
    
    masks = {
        "Global": np.ones_like(y_true, dtype=bool),
        "Peace (is_alert=0)": (is_alert_state == 0),
        "Alert (is_alert=1)": (is_alert_state == 1)
    }
    
    print(f"\n{'='*60}")
    print(f"ЗВІТ: {model_name} (Threshold = {threshold})")
    print(f"{'='*60}")
    
    for state, mask in masks.items():
        if mask.sum() == 0:
            print(f"\n--- {state} ---")
            print("Немає даних для оцінки.")
            continue
            
        y_t = y_true[mask]
        y_p = y_pred[mask]
        
        print(f"\n>>> {state} (Всього: {mask.sum()}) <<<")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_t, y_p, labels=[0, 1]))
        
        print("\nClassification Report:")
        print(classification_report(y_t, y_p, labels=[0, 1], zero_division=0))


def display_side_by_side_reports(y_true, predictions_dict, is_alert_state, threshold=0.5):
    """
    Виводить текстові репорти для кількох моделей поруч (в ряд) за допомогою HTML/CSS.
    """
    y_true = np.asarray(y_true)
    is_alert_state = np.asarray(is_alert_state)
    
    masks = {
        "Global": np.ones_like(y_true, dtype=bool),
        "Peace (is_alert=0)": (is_alert_state == 0),
        "Alert (is_alert=1)": (is_alert_state == 1)
    }
    
    # Створюємо контейнер Flexbox для горизонтального розташування
    html = f'<div style="display: flex; flex-direction: row; gap: 15px; justify-content: space-between; width: 100%;">'
    
    for model_name, y_prob in predictions_dict.items():
        y_prob = np.asarray(y_prob)
        y_pred = (y_prob >= threshold).astype(int)
        
        # Колонка для кожної моделі
        col_html = f'<div style="flex: 1; min-width: 0; border: 1px solid #ccc; padding: 15px; border-radius: 8px; background-color: rgba(128,128,128,0.05);">'
        col_html += f'<h3 style="text-align: center; margin-top: 0; color: #333;">{model_name}<br><small>(Threshold = {threshold})</small></h3>'
        
        for state, mask in masks.items():
            if mask.sum() == 0:
                continue
                
            y_t = y_true[mask]
            y_p = y_pred[mask]
            
            cm = confusion_matrix(y_t, y_p, labels=[0, 1])
            cr = classification_report(y_t, y_p, labels=[0, 1], zero_division=0)
            
            # Форматуємо Confusion Matrix вручну для рівного відступу
            cm_text = f"[[{cm[0,0]:>5} {cm[0,1]:>5}]\n [{cm[1,0]:>5} {cm[1,1]:>5}]]"
            
            col_html += f'<div style="font-weight: bold; margin-bottom: 5px; color: #0056b3;">>>> {state} (N={mask.sum()}) <<<</div>'
            col_html += f'<b>Confusion Matrix:</b>'
            col_html += f'<pre style="margin-top: 5px; margin-bottom: 10px; padding: 5px; background: none; border: none; font-size: 11px;">{cm_text}</pre>'
            
            col_html += f'<b>Classification Report:</b>'
            col_html += f'<pre style="margin-top: 5px; padding: 5px; background: none; border: none; font-size: 11px;">{cr}</pre>'
            
            # Додаємо роздільник між станами (але не після останнього)
            if state != list(masks.keys())[-1]:
                col_html += '<hr style="border: 0; border-top: 1px dashed #bbb; margin: 15px 0;">'
            
        col_html += '</div>'
        html += col_html
        
    html += '</div>'
    display(HTML(html))

def evaluate_on_test(model, df_test: pd.DataFrame, target_col='target_60_120m', threshold=0.5):
    """
    Оцінює навчену модель на тестовій вибірці (ТІЛЬКИ xx:00) 
    та виводить горизонтальний звіт: Загалом | Місяць 1 | Місяць 2.
    """
    print(f"\n{'='*80}")
    print(f"🚀 ОЦІНКА НА ТЕСТОВІЙ ВИБІРЦІ (Threshold = {threshold})")
    print(f"{'='*80}")
    
    # 1. Фільтрація (Тільки хвилина 0)
    mask_common = (df_test['timestamp'].dt.minute == 0)
    df_eval = df_test[mask_common].copy()
    y_true = df_eval[target_col].values
    is_alert_state = df_eval['is_alert'].values
    
    # 2. Отримання прогнозів
    print(f"Розмір тестової вибірки для оцінки: {len(df_eval)} рядків")
    y_prob = model.predict_proba(df_eval)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    # 3. Виділення місяців
    df_eval['month_str'] = df_eval['timestamp'].dt.strftime('%Y-%m')
    months = sorted(df_eval['month_str'].unique())
    
    # Створюємо словник зрізів
    slices = {'All Test': np.ones(len(df_eval), dtype=bool)}
    for m in months:
        slices[f"Month: {m}"] = (df_eval['month_str'] == m).values
        
    states = {
        "Global": lambda mask: mask,
        "Peace (is_alert=0)": lambda mask: mask & (is_alert_state == 0),
        "Alert (is_alert=1)": lambda mask: mask & (is_alert_state == 1)
    }
    
    # 4. Рендеринг HTML-таблиці
    # flex: 1 1 0 змушує колонки бути строго однакової ширини без розтягування
    html = f'<div style="display: flex; flex-direction: row; gap: 10px; width: 100%; font-family: monospace;">'
    
    for slice_name, slice_mask in slices.items():
        col_html = f'<div style="flex: 1 1 0; border: 1px solid #ccc; padding: 10px; border-radius: 8px; background-color: rgba(128,128,128,0.05);">'
        
        bg_color = "#e6f2ff" if slice_name == 'All Test' else "#f0f0f0"
        col_html += f'<h3 style="text-align: center; margin-top: 0; padding: 5px; background-color: {bg_color}; border-radius: 4px; color: #333;">{slice_name}</h3>'
        
        for state_name, state_func in states.items():
            state_mask = state_func(slice_mask)
            
            if state_mask.sum() == 0 or len(np.unique(y_true[state_mask])) < 2:
                continue
                
            y_t = y_true[state_mask]
            y_p = y_pred[state_mask]
            y_pr = y_prob[state_mask]
            
            cm = confusion_matrix(y_t, y_p, labels=[0, 1])
            cr = classification_report(y_t, y_p, labels=[0, 1], zero_division=0)
            ap = average_precision_score(y_t, y_pr)
            ll = log_loss(y_t, y_pr, labels=[0, 1])
            
            cm_text = f"[[{cm[0,0]:>5} {cm[0,1]:>5}]\n [{cm[1,0]:>5} {cm[1,1]:>5}]]"
            
            col_html += f'<div style="font-weight: bold; margin-top: 10px; margin-bottom: 5px; color: #0056b3;">>>> {state_name} (N={state_mask.sum()}) <<<</div>'
            col_html += f'<b>Metrics:</b> <span style="color: #d9534f;">AP={ap:.3f} | LogLoss={ll:.3f}</span>'
            
            # overflow-x: auto ТІЛЬКИ для тексту репорту, щоб не рвати колонку
            col_html += f'<pre style="margin: 5px 0; padding: 0; background: none; border: none; font-size: 11px; white-space: pre; overflow-x: auto;"><b>Confusion Matrix:</b>\n{cm_text}</pre>'
            col_html += f'<pre style="margin: 0; padding: 0; background: none; border: none; font-size: 11px; white-space: pre; overflow-x: auto;"><b>Classification Report:</b>\n{cr}</pre>'
            
            if state_name != list(states.keys())[-1]:
                 col_html += '<hr style="border: 0; border-top: 1px dashed #bbb; margin: 8px 0;">'
                 
        col_html += '</div>'
        html += col_html
        
    html += '</div>'
    display(HTML(html))
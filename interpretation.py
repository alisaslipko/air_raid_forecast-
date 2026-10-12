import shap
import matplotlib.pyplot as plt

import shap
import matplotlib.pyplot as plt

def plot_shap_summary(model, df_test, df_train, drop_cols=None, max_display=20, title="SHAP Summary Plot: Вплив ознак на ймовірність тривоги (Тест)"):
    """
    Будує SHAP Summary Plot для оцінки впливу ознак на прогноз моделі.
    
    Параметри:
    - model: об'єкт нашої моделі (MainLGBM), який містить атрибут .model
    - df_test: тестовий датафрейм
    - df_train: датафрейм, на якому навчалася модель (для синхронізації типів category)
    - drop_cols: список колонок для видалення (не фічей). За замовчуванням ['timestamp', 'target_60_120m']
    - max_display: кількість топових фічей для відображення
    - title: заголовок графіка
    """
    if drop_cols is None:
        drop_cols = ['timestamp', 'target_60_120m', 'target_15_30m']
        
    # 1. Готуємо дані для SHAP
    cols_to_drop = [col for col in df_test.columns if col in drop_cols]
    X_test_shap = df_test.drop(columns=cols_to_drop).copy()

    # === ФІКС PANDAS CATEGORY ДЛЯ SHAP ===
    # Використовуємо ._Booster, щоб дістати pandas_categorical
    cat_cols = getattr(model.model._Booster, 'pandas_categorical', None)
    if cat_cols is not None:
        for col in X_test_shap.select_dtypes(include=['object', 'category']).columns:
            if col in df_train.columns:
                X_test_shap[col] = X_test_shap[col].astype(df_train[col].dtype)
    # =======================================

    # 2. Ініціалізуємо SHAP Tree Explainer
    print("⏳ Рахуємо SHAP-значення (це може зайняти хвилину)...")
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X_test_shap)

    # Вибір правильного масиву для класу 1 (якщо SHAP повернув список)
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    # 3. Будуємо графік
    plt.figure(figsize=(10, 8))
    plt.title(title, fontsize=14)

    shap.summary_plot(
        shap_values_to_plot, 
        X_test_shap, 
        max_display=max_display, 
        show=False
    )

    plt.tight_layout()
    plt.show()
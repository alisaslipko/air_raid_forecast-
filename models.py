import time
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import lightgbm as lgb

class BaseModel:
    def __init__(self):
        self.train_time_sec = 0.0
        self.inference_time_sec = 0.0

    def save(self, path: str):
        joblib.dump(self, path)
        print(f"Модель збережено у: {path}")

    @classmethod
    def load(cls, path: str):
        print(f"Завантаження моделі з: {path}")
        return joblib.load(path)

    def get_timings(self) -> dict:
        """Повертає зафіксований час навчання та останнього передбачення."""
        return {
            "train_time_sec": self.train_time_sec,
            "inference_time_sec": self.inference_time_sec
        }

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Універсальна підготовка даних перед навчанням/інференсом.
        Видаляє службові колонки (якщо вони випадково потрапили в X) 
        та переводить 'oblast' у категоріальний тип.
        """
        X_out = X.copy()
        
        # Видаляємо службові колонки, якщо вони є
        cols_to_drop = ['timestamp', 'target_60_120m']
        existing_cols_to_drop = [c for c in cols_to_drop if c in X_out.columns]
        if existing_cols_to_drop:
            X_out = X_out.drop(columns=existing_cols_to_drop)
            
        # Категоріальний тип для LightGBM
        if 'oblast' in X_out.columns:
            X_out['oblast'] = X_out['oblast'].astype('category')
            
        return X_out


class NaiveBaseline(BaseModel):
    def __init__(self):
        super().__init__()

    def fit(self, X: pd.DataFrame, y=None):
        start_time = time.time()
        
        # Наївна модель не потребує реального навчання
        # Ми просто імітуємо інтерфейс
        
        self.train_time_sec = time.time() - start_time
        print(f"[NaiveBaseline] 'Навчання' завершено за {self.train_time_sec:.6f} сек.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        start_time = time.time()
        
        if 'is_alert' not in X.columns:
            raise ValueError("Колонка 'is_alert' є обов'язковою для NaiveBaseline")
            
        # Якщо зараз є тривога (1), ймовірність таргету = 1.0. Інакше = 0.0
        prob_1 = X['is_alert'].astype(float).values
        prob_0 = 1.0 - prob_1
        
        result = np.vstack([prob_0, prob_1]).T
        
        self.inference_time_sec = time.time() - start_time
        return result
        
    def get_feature_importance(self) -> pd.DataFrame:
        """Наївна модель не має ваг ознак, тому повертаємо порожній DataFrame."""
        return pd.DataFrame(columns=['Feature', 'Importance'])




class TreeBaseline(BaseModel):
    def __init__(self, max_depth=5, random_state=42):
        super().__init__()
        
        # Налаштовуємо LightGBM для роботи в режимі "одного дерева"
        self.model = LGBMClassifier(
            n_estimators=1,
            max_depth=max_depth,
            learning_rate=1.0,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1  # Вимикаємо зайві системні логи LightGBM
        )
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        start_time = time.time()
        
        # Підготовка ознак (тут 'oblast' стає 'category')
        X_prep = self._prepare_features(X)
        self.feature_names_ = X_prep.columns.tolist()
        
        self.model.fit(X_prep, y)
        
        self.train_time_sec = time.time() - start_time
        print(f"[TreeBaseline] Навчання 1 дерева (max_depth={self.model.max_depth}) завершено за {self.train_time_sec:.4f} сек.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        start_time = time.time()
        
        X_prep = self._prepare_features(X)
        result = self.model.predict_proba(X_prep)
        
        self.inference_time_sec = time.time() - start_time
        return result
        
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Повертає важливість ознак для цього дерева.
        За замовчуванням LightGBM повертає кількість сплітів (importance_type='split').
        """
        if self.feature_names_ is None:
            raise ValueError("Модель ще не навчена! Спочатку викличте fit().")
            
        df_imp = pd.DataFrame({
            'Feature': self.feature_names_,
            'Importance': self.model.feature_importances_
        })
        
        # Залишаємо тільки ті фічі, які дерево реально використало (>0), і сортуємо
        df_imp = df_imp[df_imp['Importance'] > 0].sort_values(by='Importance', ascending=False).reset_index(drop=True)
        return df_imp


class MainLGBM(BaseModel):
    def __init__(self, params=None, early_stopping_rounds=50):
        super().__init__()
        self.early_stopping_rounds = early_stopping_rounds
        
        # Дефолтні параметри. Важливість ('gain') задаємо одразу для отримання коректних ваг.
        default_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'random_state': 42,
            'n_jobs': -1,
            'importance_type': 'gain' 
        }
        if params:
            default_params.update(params)
            
        self.model = lgb.LGBMClassifier(**default_params)
        self.best_iteration = None
        self.feature_names_ = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        start_time = time.time()
        
        # Відбираємо фічі (залишаємо is_alert, видаляємо лише час та таргети!)
        drop_cols = [col for col in X_train.columns if col in ['timestamp', 'target_60_120m']]
        X_tr = X_train.drop(columns=drop_cols)

        self.feature_names_ = X_tr.columns.tolist()
        
        # Базові аргументи для fit
        fit_kwargs = {}
        
        # Якщо валідаційні дані передані — вмикаємо early stopping
        if X_val is not None and y_val is not None:
            X_v = X_val.drop(columns=drop_cols)
            fit_kwargs['eval_set'] = [(X_v, y_val)]
            
            if getattr(self, 'early_stopping_rounds', None) is not None:
                fit_kwargs['callbacks'] = [
                    lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False)
                ]
                
        # Навчаємо модель
        self.model.fit(X_tr, y_train, **fit_kwargs)
        
        # Зберігаємо таймінги
        self.train_time_sec = time.time() - start_time
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        start_time = time.time()
        
        X_prep = self._prepare_features(X)
        
        # Передаємо num_iteration лише якщо він валідний
        kwargs = {}
        if self.best_iteration is not None:
            kwargs['num_iteration'] = self.best_iteration
            
        result = self.model.predict_proba(X_prep, **kwargs)
        
        self.inference_time_sec = time.time() - start_time
        return result
        
    def get_feature_importance(self) -> pd.DataFrame:
        """Повертає важливість ознак (Gain) у відсортованому вигляді."""
        if self.feature_names_ is None:
            raise ValueError("Модель ще не навчена!")
            
        df_imp = pd.DataFrame({
            'Feature': self.feature_names_,
            'Importance': self.model.feature_importances_
        })
        
        # Залишаємо фічі з важливістю > 0 та сортуємо
        df_imp = df_imp[df_imp['Importance'] > 0].sort_values(by='Importance', ascending=False).reset_index(drop=True)
        return df_imp
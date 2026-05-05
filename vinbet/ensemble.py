import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency handled by caller
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - optional dependency handled by caller
    CatBoostClassifier = None


@dataclass
class EnsembleWeights:
    lgbm: float = 0.55
    xgb: float = 0.30
    cat: float = 0.15

    def normalized(self) -> Dict[str, float]:
        raw = {'lgbm': float(self.lgbm), 'xgb': float(self.xgb), 'cat': float(self.cat)}
        total = sum(raw.values())
        if total <= 0:
            raise ValueError('Ensemble weights must sum to a positive value.')
        return {k: v / total for k, v in raw.items()}


class PlattCalibrator:
    """One-dimensional logistic calibration for model probabilities."""

    def __init__(self):
        self._model = LogisticRegression(solver='lbfgs', max_iter=1000)

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> 'PlattCalibrator':
        x = np.asarray(raw_probs).reshape(-1, 1)
        self._model.fit(x, y_true)
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        x = np.asarray(raw_probs).reshape(-1, 1)
        return self._model.predict_proba(x)[:, 1]


class WeightedEnsembleClassifier:
    def __init__(self, weights: Optional[EnsembleWeights] = None):
        self.weights = (weights or EnsembleWeights()).normalized()
        self.models: Dict[str, object] = {}
        self.calibrators: Dict[str, PlattCalibrator] = {}

    @staticmethod
    def _check_dependencies(use_catboost: bool = True):
        if XGBClassifier is None:
            raise ImportError('xgboost is not installed. Add xgboost to dependencies and install.')
        if use_catboost and CatBoostClassifier is None:
            raise ImportError('catboost is not installed. Add catboost to dependencies and install.')

    @staticmethod
    def _as_catboost_frame(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        out = df.copy()
        for col in cat_cols:
            out[col] = out[col].astype(str)
        return out

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        categorical_cols: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Fits the three base models and their calibrators on validation probabilities."""
        self._check_dependencies(use_catboost=True)
        categorical_cols = categorical_cols or []

        # LightGBM anchor model (stable baseline)
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        lgbm = lgb.train(
            {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'learning_rate': 0.03,
                'num_leaves': 31,
                'feature_fraction': 0.75,
                'bagging_fraction': 0.75,
                'bagging_freq': 5,
                'min_child_samples': 400,
                'lambda_l1': 0.5,
                'lambda_l2': 1.0,
                'verbosity': -1,
            },
            train_data,
            num_boost_round=1200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=80), lgb.log_evaluation(period=0)],
        )
        lgbm_raw = lgbm.predict(X_val, num_iteration=lgbm.best_iteration)

        # XGBoost model with conservative anti-overfit setup
        xgb = XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=8,
            subsample=0.70,
            colsample_bytree=0.70,
            reg_alpha=1.0,
            reg_lambda=8.0,
            gamma=1.0,
            n_estimators=600,
            random_state=42,
            tree_method='hist',
            early_stopping_rounds=60,
        )
        xgb.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        xgb_raw = xgb.predict_proba(X_val)[:, 1]

        # CatBoost model leverages categorical regime columns.
        cat_train = self._as_catboost_frame(X_train, categorical_cols)
        cat_val = self._as_catboost_frame(X_val, categorical_cols)
        cat = CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='AUC',
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=10.0,
            random_strength=1.5,
            bagging_temperature=1.0,
            iterations=800,
            od_type='Iter',
            od_wait=80,
            verbose=False,
            random_seed=42,
        )
        cat.fit(
            cat_train,
            y_train,
            eval_set=(cat_val, y_val),
            cat_features=categorical_cols,
            use_best_model=True,
        )
        cat_raw = cat.predict_proba(cat_val)[:, 1]

        self.models = {'lgbm': lgbm, 'xgb': xgb, 'cat': cat}
        self.calibrators = {
            'lgbm': PlattCalibrator().fit(lgbm_raw, y_val.values),
            'xgb': PlattCalibrator().fit(xgb_raw, y_val.values),
            'cat': PlattCalibrator().fit(cat_raw, y_val.values),
        }

        return {
            'lgbm_raw': lgbm_raw,
            'xgb_raw': xgb_raw,
            'cat_raw': cat_raw,
            'ensemble_prob': self._blend_raw(lgbm_raw, xgb_raw, cat_raw),
        }

    def _predict_raw(
        self,
        X: pd.DataFrame,
        categorical_cols: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        cat_cols = categorical_cols or []
        lgbm = self.models['lgbm']
        xgb = self.models['xgb']
        cat = self.models['cat']

        lgbm_raw = lgbm.predict(X, num_iteration=getattr(lgbm, 'best_iteration', None))
        xgb_raw = xgb.predict_proba(X)[:, 1]
        cat_raw = cat.predict_proba(self._as_catboost_frame(X, cat_cols))[:, 1]
        return {'lgbm': lgbm_raw, 'xgb': xgb_raw, 'cat': cat_raw}

    def _blend_raw(self, lgbm_raw: np.ndarray, xgb_raw: np.ndarray, cat_raw: np.ndarray) -> np.ndarray:
        lgbm_cal = self.calibrators['lgbm'].transform(lgbm_raw)
        xgb_cal = self.calibrators['xgb'].transform(xgb_raw)
        cat_cal = self.calibrators['cat'].transform(cat_raw)

        return (
            self.weights['lgbm'] * lgbm_cal
            + self.weights['xgb'] * xgb_cal
            + self.weights['cat'] * cat_cal
        )

    def predict_proba(self, X: pd.DataFrame, categorical_cols: Optional[List[str]] = None) -> np.ndarray:
        raw = self._predict_raw(X, categorical_cols=categorical_cols)
        return self._blend_raw(raw['lgbm'], raw['xgb'], raw['cat'])

    def save(self, output_dir: str, metadata: Optional[Dict] = None):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self.models['lgbm'].save_model(str(out / 'lgbm_model.txt'))
        self.models['xgb'].save_model(str(out / 'xgb_model.json'))
        self.models['cat'].save_model(str(out / 'cat_model.cbm'))

        with open(out / 'calibrators.pkl', 'wb') as f:
            pickle.dump(self.calibrators, f)

        payload = {'weights': self.weights}
        if metadata:
            payload.update(metadata)
        with open(out / 'ensemble_config.json', 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, output_dir: str) -> Tuple['WeightedEnsembleClassifier', Dict]:
        cls._check_dependencies(use_catboost=True)
        out = Path(output_dir)

        with open(out / 'ensemble_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        model = cls(EnsembleWeights(**config['weights']))

        lgbm = lgb.Booster(model_file=str(out / 'lgbm_model.txt'))
        xgb = XGBClassifier()
        xgb.load_model(str(out / 'xgb_model.json'))

        cat = CatBoostClassifier()
        cat.load_model(str(out / 'cat_model.cbm'))

        with open(out / 'calibrators.pkl', 'rb') as f:
            calibrators = pickle.load(f)

        model.models = {'lgbm': lgbm, 'xgb': xgb, 'cat': cat}
        model.calibrators = calibrators
        return model, config


def split_time_series(X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8):
    split_idx = int(len(X) * train_ratio)
    return (
        X.iloc[:split_idx],
        X.iloc[split_idx:],
        y.iloc[:split_idx],
        y.iloc[split_idx:],
        split_idx,
    )

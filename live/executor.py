import json
import lightgbm as lgb
import logging
from pathlib import Path

import pandas as pd

from aightbet.ensemble import WeightedEnsembleClassifier

logger = logging.getLogger(__name__)


class Executor:
    """Position-sizing executor driven by model confidence."""

    def __init__(
        self,
        model_type="auto",
        model_path="lgbm_model.txt",
        feature_cols_path="feature_columns.json",
        best_params_path="params_tuned.json",
        ensemble_dir="models/ensemble_v1",
    ):
        """
        Initialize executor with a trained model.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved LightGBM model.
        feature_cols_path : str
            Path to JSON file with list of feature column names.
        best_params_path : str
            Path to JSON file with best parameters (including threshold, stop_loss, take_profit).
        """
        self.position_units = 0  # Signed units: negative short, positive long.
        self.avg_entry_price = None
        self.forced_exit_active = False
        self.model_type = None
        self.categorical_cols = []
        
        ensemble_config_path = Path(ensemble_dir) / "ensemble_config.json"
        selected_model_type = model_type
        if selected_model_type == "auto":
            selected_model_type = "ensemble" if ensemble_config_path.exists() else "lgbm"

        self.best_params = {}

        if selected_model_type == "ensemble":
            ensemble, ensemble_config = WeightedEnsembleClassifier.load(ensemble_dir)
            self.model = ensemble
            self.model_type = "ensemble"
            self.feature_cols = ensemble_config["feature_columns"]
            self.categorical_cols = ensemble_config.get("categorical_columns", [])
            self.threshold = float(ensemble_config.get("threshold", 0.55))
            self.stop_loss_pct = float(ensemble_config.get("stop_loss", 0.015))
            self.take_profit_pct = float(ensemble_config.get("take_profit", 0.025))
            logger.info("Loaded ensemble model from %s", ensemble_dir)
            logger.info("Loaded %d feature columns (%d categorical)", len(self.feature_cols), len(self.categorical_cols))
        elif selected_model_type == "lgbm":
            self.model = lgb.Booster(model_file=model_path)
            self.model_type = "lgbm"
            logger.info("Loaded model from %s", model_path)

            with open(feature_cols_path, "r", encoding="utf-8") as f:
                self.feature_cols = json.load(f)
            logger.info("Loaded %d feature columns", len(self.feature_cols))
            self.threshold = 0.55
            self.stop_loss_pct = 0.015
            self.take_profit_pct = 0.025
        else:
            raise ValueError("model_type must be one of: auto, lgbm, ensemble")

        # Optional overrides from params_tuned.json (mainly useful for LGBM mode)
        try:
            with open(best_params_path, "r", encoding="utf-8") as f:
                self.best_params = json.load(f)
        except FileNotFoundError:
            logger.info("best_params file not found at %s, using model defaults", best_params_path)

        self.threshold = float(self.best_params.get("threshold", self.threshold))
        self.stop_loss_pct = float(self.best_params.get("stop_loss", self.stop_loss_pct))
        self.take_profit_pct = float(self.best_params.get("take_profit", self.take_profit_pct))
        self.max_position_units = int(self.best_params.get("max_position_units", 3))
        self.confidence_step = float(self.best_params.get("confidence_step", 0.08))
        logger.info("Model type: %s", self.model_type)
        logger.info("Threshold set to %.4f", self.threshold)
        logger.info("Stop Loss: %.2f%%, Take Profit: %.2f%%", self.stop_loss_pct * 100, self.take_profit_pct * 100)
        logger.info(
            "Position sizing enabled: max_units=%d, confidence_step=%.3f",
            self.max_position_units,
            self.confidence_step,
        )

    def _predict_prob(self, features):
        if self.model_type == "ensemble":
            row = {}
            for col in self.feature_cols:
                val = features[col]
                if col in self.categorical_cols:
                    row[col] = int(val)
                else:
                    row[col] = float(val)
            X = pd.DataFrame([row], columns=self.feature_cols)
            return float(self.model.predict_proba(X, categorical_cols=self.categorical_cols)[0])

        X = [[features[col] for col in self.feature_cols]]
        return float(self.model.predict(X)[0])

    def _check_risk_exit(self, current_price):
        """Return True when SL/TP requires flattening."""
        if self.position_units == 0 or self.avg_entry_price is None:
            return None
        if self.position_units > 0:
            if current_price <= self.avg_entry_price * (1 - self.stop_loss_pct):
                return "STOP_LOSS"
            if current_price >= self.avg_entry_price * (1 + self.take_profit_pct):
                return "TAKE_PROFIT"
        else:
            if current_price >= self.avg_entry_price * (1 + self.stop_loss_pct):
                return "STOP_LOSS"
            if current_price <= self.avg_entry_price * (1 - self.take_profit_pct):
                return "TAKE_PROFIT"
        return None

    def _desired_position_units(self, prob_up):
        buy_conf = prob_up
        sell_conf = 1.0 - prob_up
        side_conf = max(buy_conf, sell_conf)
        if side_conf < self.threshold:
            return 0, buy_conf, sell_conf, side_conf

        if self.confidence_step > 0:
            extra_steps = int((side_conf - self.threshold) / self.confidence_step)
        else:
            extra_steps = 0
        units = max(1, min(self.max_position_units, 1 + extra_steps))

        if buy_conf > sell_conf:
            return units, buy_conf, sell_conf, side_conf
        if sell_conf > buy_conf:
            return -units, buy_conf, sell_conf, side_conf
        return 0, buy_conf, sell_conf, side_conf

    def _apply_fill(self, side, current_price):
        """Update internal position and average entry after a simulated fill."""
        if side == "BUY":
            if self.position_units >= 0:
                prior_units = self.position_units
                self.position_units += 1
                if prior_units == 0 or self.avg_entry_price is None:
                    self.avg_entry_price = current_price
                else:
                    self.avg_entry_price = ((self.avg_entry_price * prior_units) + current_price) / self.position_units
            else:
                self.position_units += 1
                if self.position_units == 0:
                    self.avg_entry_price = None
        elif side == "SELL":
            if self.position_units <= 0:
                prior_units = abs(self.position_units)
                self.position_units -= 1
                if prior_units == 0 or self.avg_entry_price is None:
                    self.avg_entry_price = current_price
                else:
                    new_units = abs(self.position_units)
                    self.avg_entry_price = ((self.avg_entry_price * prior_units) + current_price) / new_units
            else:
                self.position_units -= 1
                if self.position_units == 0:
                    self.avg_entry_price = None

    def generate_signal(self, features, current_price):
        """
        Generate BUY/SELL/HOLD signal based on model prediction.
        Only generates signals when confidence (probability) exceeds threshold.
        
        Parameters:
        -----------
        features : dict
            Dictionary of feature name -> value pairs from FeatureEngine.
        current_price : float
            Current market price for position tracking.
            
        Returns:
        --------
        str or None
            "BUY", "SELL", "HOLD", "EXIT_SL", "EXIT_TP", or None.
        """
        if features is None:
            return None

        payload_features = dict(features)

        # Fill missing categorical features with unknown bucket if not produced by the stream.
        missing = [col for col in self.feature_cols if col not in payload_features]
        for col in missing:
            if col in self.categorical_cols:
                payload_features[col] = -1

        missing = [col for col in self.feature_cols if col not in payload_features]
        if missing:
            logger.warning(f"Missing features: {missing}")
            return None

        # Risk management has priority over model sizing.
        risk_exit = self._check_risk_exit(current_price)
        if risk_exit and self.position_units != 0:
            side = "SELL" if self.position_units > 0 else "BUY"
            self._apply_fill(side, current_price)
            logger.info(
                "%s triggered at %.2f | action=%s | remaining_units=%d",
                risk_exit,
                current_price,
                side,
                self.position_units,
            )
            if self.position_units == 0:
                self.forced_exit_active = False
            else:
                self.forced_exit_active = True
            return side

        if self.forced_exit_active and self.position_units != 0:
            side = "SELL" if self.position_units > 0 else "BUY"
            self._apply_fill(side, current_price)
            logger.info("Continuing forced exit with %s | remaining_units=%d", side, self.position_units)
            if self.position_units == 0:
                self.forced_exit_active = False
            return side

        # Predict probability of up move and derive target position.
        prob = self._predict_prob(payload_features)
        target_units, buy_confidence, sell_confidence, side_confidence = self._desired_position_units(prob)

        if target_units > self.position_units:
            self._apply_fill("BUY", current_price)
            logger.info(
                "BUY step | prob_up=%.4f conf=%.4f target=%d now=%d avg_entry=%.2f",
                prob,
                side_confidence,
                target_units,
                self.position_units,
                self.avg_entry_price or 0.0,
            )
            return "BUY"
        if target_units < self.position_units:
            self._apply_fill("SELL", current_price)
            logger.info(
                "SELL step | prob_up=%.4f conf=%.4f target=%d now=%d avg_entry=%.2f",
                prob,
                side_confidence,
                target_units,
                self.position_units,
                self.avg_entry_price or 0.0,
            )
            return "SELL"

        logger.info(
            "HOLD | prob_up=%.4f buy_conf=%.4f sell_conf=%.4f target=%d now=%d",
            prob,
            buy_confidence,
            sell_confidence,
            target_units,
            self.position_units,
        )
        return "HOLD"

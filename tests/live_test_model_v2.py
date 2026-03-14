from sklearn.ensemble import RandomForestClassifier

from autoquant_cli.quant.model_base import AutoQuantModel


class LiveTestModelV2(AutoQuantModel):
    def create_features(self, frame):
        frame = frame.copy()
        close = frame["close"].astype(float)
        volume = frame["volume"].astype(float)
        frame["feature_close"] = close
        frame["feature_return_1"] = close.pct_change().fillna(0.0)
        frame["feature_return_3"] = close.pct_change(3).fillna(0.0)
        frame["feature_range"] = ((frame["high"].astype(float) - frame["low"].astype(float)) / close.replace(0, 1)).fillna(0.0)
        frame["feature_volume_change"] = volume.pct_change().replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        frame["feature_ma_gap"] = ((close / close.rolling(8).mean()) - 1.0).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        frame["target"] = (close.shift(-1) > close).astype(int)
        frame = frame.iloc[:-1].reset_index(drop=True)
        return frame, [
            "feature_close",
            "feature_return_1",
            "feature_return_3",
            "feature_range",
            "feature_volume_change",
            "feature_ma_gap",
        ]

    def get_hyperparameter_candidates(self):
        return {
            "n_estimators": [100, 200, 300],
            "max_depth": (3, 5),
            "min_samples_leaf": [2, 3, 5],
        }

    def fit(self, x_train, y_train, hyperparams):
        self.model = RandomForestClassifier(
            n_estimators=int(hyperparams.get("n_estimators", 200)),
            max_depth=int(hyperparams.get("max_depth", 4)),
            min_samples_leaf=int(hyperparams.get("min_samples_leaf", 3)),
            random_state=42,
        )
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test).tolist()

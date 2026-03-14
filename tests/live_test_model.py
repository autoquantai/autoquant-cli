from sklearn.linear_model import LogisticRegression

from autoquant_cli.quant.model_base import AutoQuantModel


class LiveTestModel(AutoQuantModel):
    def create_features(self, frame):
        frame = frame.copy()
        frame["feature_close"] = frame["close"].astype(float)
        frame["feature_return"] = frame["close"].pct_change().fillna(0.0)
        frame["feature_volume"] = frame["volume"].pct_change().replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        frame["target"] = (frame["close"].shift(-1) > frame["close"]).astype(int)
        frame = frame.iloc[:-1].reset_index(drop=True)
        return frame, ["feature_close", "feature_return", "feature_volume"]

    def fit(self, x_train, y_train, hyperparams):
        self.model = LogisticRegression(max_iter=200)
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test).tolist()

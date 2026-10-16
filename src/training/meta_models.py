"""Interpretable candidate models and training-only probability calibration."""
import numpy as np


class SklearnMetaModel:
    def __init__(self, estimator, mean, std, columns, kind):
        self.estimator, self.mean, self.std = estimator, mean, std
        self.feature_columns, self.kind = list(columns), kind

    def matrix(self, frame):
        return (frame.reindex(columns=self.feature_columns).replace([np.inf, -np.inf], np.nan).fillna(0.).to_numpy(dtype=float) - self.mean) / self.std

    def predict_proba(self, frame):
        return self.estimator.predict_proba(self.matrix(frame))[:, 1] if len(frame) else np.array([])

    def manifest(self):
        importance = (self.estimator.coef_[0] if self.kind == 'logistic' else self.estimator.feature_importances_)
        return {'kind': self.kind, 'feature_columns': self.feature_columns,
                'interpretability': {'type': 'standardized_coefficients' if self.kind == 'logistic' else 'impurity_importance',
                                     'values': dict(zip(self.feature_columns, importance.tolist()))},
                'preprocessing': {'fit_scope': 'training_rows_only', 'mean': self.mean.tolist(), 'std': self.std.tolist()}}


class MetaModelBundle:
    def __init__(self, base, calibrator=None, method='none', audit=None):
        self.base, self.calibrator, self.method = base, calibrator, method
        self.audit = audit or {}
        self.kind = base.kind
        self.feature_columns = base.feature_columns
        self.decision_threshold = None

    @property
    def probability(self):
        return self.base.probability

    def predict_proba(self, frame):
        p = self.base.predict_proba(frame)
        if self.calibrator is None or not len(p):
            return p
        if self.method == 'platt':
            return self.calibrator.predict_proba(logits(p).reshape(-1, 1))[:, 1]
        return self.calibrator.predict(p)

    def manifest(self):
        result = {**self.base.manifest(), **self.audit, 'decision_threshold': self.decision_threshold}
        if self.calibrator is not None:
            result['calibrator'] = ({'coefficient': self.calibrator.coef_.tolist(), 'intercept': self.calibrator.intercept_.tolist()}
                                    if self.method == 'platt' else
                                    {'x': self.calibrator.X_thresholds_.tolist(), 'y': self.calibrator.y_thresholds_.tolist()})
        return result


def logits(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def fit_calibrator(method, probability, truth, weights):
    if method == 'platt':
        from sklearn.linear_model import LogisticRegression
        estimator = LogisticRegression(max_iter=1000, random_state=0)
        estimator.fit(logits(probability).reshape(-1, 1), truth, sample_weight=weights)
    elif method == 'isotonic':
        from sklearn.isotonic import IsotonicRegression
        estimator = IsotonicRegression(out_of_bounds='clip')
        estimator.fit(probability, truth, sample_weight=weights)
    else:
        raise ValueError(f'Unknown calibration method: {method}')
    return estimator

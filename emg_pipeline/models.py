from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, ClassifierMixin
import logging

logger = logging.getLogger(__name__)

class EMGModel(BaseEstimator, ClassifierMixin):
    """
    Base wrapper for EMG models.
    """
    def __init__(self, model_type='lda'):
        self.model_type = model_type
        self.model = self._get_model(model_type)
        
    def _get_model(self, model_type):
        if model_type == 'lda':
            return LinearDiscriminantAnalysis()
        elif model_type == 'lr':
            return LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced')
        elif model_type == 'svm':
            return LinearSVC(max_iter=1000, multi_class='ovr', class_weight='balanced')
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def fit(self, X, y):
        logger.info(f"Training {self.model_type} on {X.shape[0]} samples...")
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # SVM doesn't support probability by default
            # We could use calibrator, but for now just return decision function or nothing
            logger.warning(f"{self.model_type} does not support predict_proba")
            return None

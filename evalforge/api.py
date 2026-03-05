import pandas as pd
from evalforge.metrics import calculate_metrics
from evalforge.mismatch import detect_confidence_accuracy_mismatch
from evalforge.fragility import calculate_adversarial_fragility
from evalforge.drift import detect_drift
from evalforge.health_score import compute_health_score
from evalforge.report_card import generate_report_card
from evalforge.fairness import evaluate_fairness

class ModelAuditor:
    """
    An object-oriented API for evaluating ML models in Python.
    Because making users drop into bash just to evaluate a model is slightly tyrannical :)
    """
    
    def __init__(self, model, target_col="target", task_type="classification"):
        """
        Initializes the auditor.
        
        Args:
            model: A trained scikit-learn compatible model (must have `.predict()`).
            target_col (str): The name of the target variable column in the datasets.
        """
        if not hasattr(model, "predict"):
            raise TypeError("Provided model does not have a predict() method. What are you trying to test? :(")
            
        self.model = model
        self.target_col = target_col
        self.task_type = task_type
        self.health_score_data_ = None
        self.mismatch_data_ = None
        self.fragility_data_ = None
        self.drift_data_ = None
        self.accuracy_ = None
        self.fairness_data_ = None
        
    def _extract_xy(self, df):
        """Helper to safely split out target from features."""
        if isinstance(df, pd.DataFrame):
            if self.target_col not in df.columns:
                raise ValueError(f"Target column '{self.target_col}' missing from dataframe.")
            y = df[self.target_col]
            X = df.drop(columns=[self.target_col])
        else:
            raise TypeError("Inputs must be pandas DataFrames.")
        return X, y

    def evaluate(self, df_test, df_train=None, run_fragility=True, sensitive_col=None):
        """
        Runs the full suite of EvalForge diagnostics on the model.
        
        Args:
            df_test (pd.DataFrame): The test dataset to evaluate against.
            df_train (pd.DataFrame, optional): The training dataset, needed to detect drift.
            run_fragility (bool): Whether to run adversarial perturbation tests (can be slow).
            sensitive_col (str, optional): The column to check for demographic bias.
            
        Returns:
            dict: The final Health Score dictionary (so you don't have to parse text).
        """
        X_test, y_true = self._extract_xy(df_test)
        
        # Base predictions
        y_pred = self.model.predict(X_test)
        
        y_prob = None
        if hasattr(self.model, "predict_proba"):
            try:
                y_prob = self.model.predict_proba(X_test)
            except Exception:
                pass # Model said no
                
        # 1. Base Metrics
        metrics = calculate_metrics(y_true, y_pred, y_prob, task_type=self.task_type)
        if self.task_type == "regression":
            # For regression, we might proxy accuracy using R2 bound between 0-1 for health score
            self.accuracy_ = max(0.0, metrics.get("r2_score", 0.0))
        else:
            self.accuracy_ = metrics.get("accuracy", 0.0)
        
        # 2. Calibration Mismatch
        if y_prob is not None:
            self.mismatch_data_ = detect_confidence_accuracy_mismatch(y_true, y_pred, y_prob)
            
        # 3. Fragility
        if run_fragility:
            self.fragility_data_ = calculate_adversarial_fragility(self.model, X_test, y_true)
            
        # 4. Drift detection
        if df_train is not None:
            X_train, _ = self._extract_xy(df_train)
            self.drift_data_ = detect_drift(X_train, X_test)
            
        # 4.5. Fairness constraints
        bias_penalty = 0.0
        if sensitive_col is not None:
             self.fairness_data_ = evaluate_fairness(df_test, y_pred, sensitive_col)
             bias_penalty = self.fairness_data_.get("bias_penalty", 0.0)
            
        # 5. Bring it all together
        mismatch_rate = self.mismatch_data_["mismatch_rate"] if self.mismatch_data_ else None
        fragility_score = self.fragility_data_["fragility_score"] if self.fragility_data_ else None
        
        self.health_score_data_ = compute_health_score(
            accuracy_metric=self.accuracy_,
            mismatch_rate=mismatch_rate,
            fragility_score=fragility_score,
            drift_report=self.drift_data_,
            bias_penalty=bias_penalty
        )
        
        return self.health_score_data_

    def get_report(self):
        """
        Generates the formatted markdown report based on the last `.evaluate()` run.
        """
        if self.health_score_data_ is None:
            raise RuntimeError("You must call .evaluate() before getting the report! We can't evaluate the void.")
            
        return generate_report_card(
            self.health_score_data_, 
            self.mismatch_data_, 
            self.drift_data_, 
            self.fragility_data_
        )

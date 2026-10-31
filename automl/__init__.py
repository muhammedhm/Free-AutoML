"""Tiny free AutoML library built on FLAML + scikit-learn."""
from .trainer import run_automl
from .predict import load_model, predict

__all__ = ["run_automl", "load_model", "predict"]

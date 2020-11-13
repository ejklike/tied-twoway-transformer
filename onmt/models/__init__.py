"""Module defining models."""
from onmt.models.model_saver import build_model_saver, ModelSaver
from onmt.models.model import NMTModel
from onmt.models.classifier import Classifier, Classifier2

__all__ = ["build_model_saver", "ModelSaver", "NMTModel", "Classifier", 
           "Classifier2"]

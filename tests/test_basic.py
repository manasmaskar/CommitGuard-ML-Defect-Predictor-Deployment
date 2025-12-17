"""Basic tests for the prediction system"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import joblib
import config


def test_model_exists():
    """Test that model file exists"""
    assert os.path.exists(config.MODEL_PATH), "Model file not found"
    print("Model file exists")


def test_model_loads():
    """Test that model can be loaded"""
    model = joblib.load(config.MODEL_PATH)
    assert model is not None
    print("Model loads successfully")


def test_feature_columns_count():
    """Test that we have 21 features"""
    assert len(config.FEATURE_COLUMNS) == 21
    print("Feature count is correct (21)")


def test_prediction_shape():
    """Test prediction output shape"""
    model = joblib.load(config.MODEL_PATH)
    
    # Create dummy input
    dummy_input = {col: 10.0 for col in config.FEATURE_COLUMNS}
    features_df = pd.DataFrame([dummy_input])
    
    prediction = model.predict(features_df)
    probability = model.predict_proba(features_df)
    
    assert prediction.shape == (1,)
    assert probability.shape == (1, 2)
    print("Prediction output shapes are correct")


def test_risk_thresholds():
    """Test risk threshold configuration"""
    assert config.RISK_THRESHOLDS['HIGH'] > config.RISK_THRESHOLDS['MEDIUM']
    assert config.RISK_THRESHOLDS['MEDIUM'] > config.RISK_THRESHOLDS['LOW']
    print("Risk thresholds are properly ordered")


if __name__ == "__main__":
    print("Running tests...\n")
    test_model_exists()
    test_model_loads()
    test_feature_columns_count()
    test_prediction_shape()
    test_risk_thresholds()
    print("All tests passed!")

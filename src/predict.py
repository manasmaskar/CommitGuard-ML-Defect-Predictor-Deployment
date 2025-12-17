'''CLI Prediction Script'''

import argparse
import json
import joblib
import pandas as pd
import sys

import config 
from explain import get_decision_path, get_risk_level, get_top_features

def load_model():
    '''Load Trained Model'''
    try: 
        model = joblib.load(config.MODEL_PATH)
        return model
    except FileNotFoundError:
        print(f"Error: Model not found at {config.MODEL_PATH}")
        print("Please Run the train model! ")
        sys.exit(1)

def validate_input(features_dict):
    '''Validate the input features'''
    missing = set(config.FEATURE_COLUMNS) - set(features_dict.keys())
    if missing: 
        print(f"Error: Missing Features: {missing}")
        sys.exit(1)

    extra = set(features_dict.keys()) - set(config.FEATURE_COLUMNS)
    if extra:
        print(f"Warning: Extra features put those will be ignored: {extra}")
    return True

def predict(model, features_dict):
    '''Make Predict'''
    features_df = pd.DataFrame([features_dict])[config.FEATURE_COLUMNS]

    # Predict 
    prediction = model.predict(features_df)[0]
    proba_array = model.predict_proba(features_df)[0]  # Get array for first sample
    defect_prob = proba_array[1]
    return prediction, defect_prob, features_df

def main():
    parser = argparse.ArgumentParser(
        description = 'Predict PR Defect risk using decision tree'
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to JSON file with PR Features'
    )
    args = parser.parse_args()

    #Load input
    try:
        with open(args.input, 'r') as f:
            features = json.load(f)
    except FileNotFoundError:
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {args.input}")
        sys.exit(1)

    #Validate
    validate_input(features)

    #Load Model
    print("Loading Model ...")
    model = load_model()

    #Predict
    print("making prediction.. \n")
    prediction, defect_prob, features_df = predict(model, features)

    #Explain 
    decision_path = get_decision_path(model, features_df, config.FEATURE_COLUMNS)
    risk_level = get_risk_level(defect_prob)
    top_features = get_top_features(
        model, 
        config.FEATURE_COLUMNS,
        features_df.iloc[0].values
    )

    #Output
    print("="*60)
    print("PR Quality Gate Prediction")
    print("="*60)

    if prediction == 1:
        print("Prediction: Defect Prone")
    else:
        print("Prediction: CLEAN CODE") 
    
    print(f"Defect Probability: {defect_prob:.2%}")
    print(f"Risk Score: {int(defect_prob * 100)}/100")
    print(f"Risk Level: {risk_level}")

    print("\n" + "="*60)
    print("DECISION PATH")
    print("="*60)
    if decision_path:
        for i, rule in enumerate(decision_path, 1):
            print(f"  {i}. {rule}")
    else:
        print("  Root node (single leaf tree)")

    print("\n" + "="*60)
    print("TOP CONTRIBUTING FEATURES")
    print("="*60)
    for i, feat in enumerate(top_features, 1):
        print(f"  {i}. {feat['name']}: {feat['value']:.2f}")
        print(f"     (feature importance: {feat['importance']:.4f})")
    
    print("\n" + "="*60)

    # Return code based on risk
    if defect_prob >= config.RISK_THRESHOLDS['HIGH']:
        sys.exit(2)  # High risk
    elif defect_prob >= config.RISK_THRESHOLDS['MEDIUM']:
        sys.exit(1)  # Medium risk
    else:
        sys.exit(0)  # Low risk


if __name__ == "__main__":
    main()
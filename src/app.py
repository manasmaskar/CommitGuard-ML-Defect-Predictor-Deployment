"""
CommitGuard: PR Defect Predictor
Simple Streamlit app for portfolio demonstration
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Page config
st.set_page_config(
    page_title="CommitGuard: PR Defect Predictor",
    page_icon="🛡️",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/alternatives/recall_optimized_model.pkl')
        return model
    except:
        st.error("Model not found. Please train the model first.")
        return None

# Feature columns
FEATURE_COLUMNS = [
    'cbo', 'wmc', 'dit', 'rfc', 'lcom', 'totalMethods', 'totalFields', 
    'nosi', 'loc', 'returnQty', 'loopQty', 'comparisonsQty', 'tryCatchQty',
    'parenthesizedExpsQty', 'stringLiteralsQty', 'numbersQty', 
    'assignmentsQty', 'mathOperationsQty', 'variablesQty', 
    'maxNestedBlocks', 'uniqueWordsQty'
]

# Example PRs
EXAMPLES = {
    "🟢 Low Risk PR - Simple Code": {
        "cbo": 2, "wmc": 8, "dit": 1, "rfc": 12, "lcom": 15,
        "totalMethods": 5, "totalFields": 3, "nosi": 0, "loc": 45,
        "returnQty": 3, "loopQty": 1, "comparisonsQty": 5, "tryCatchQty": 0,
        "parenthesizedExpsQty": 2, "stringLiteralsQty": 8, "numbersQty": 10,
        "assignmentsQty": 12, "mathOperationsQty": 3, "variablesQty": 8,
        "maxNestedBlocks": 2, "uniqueWordsQty": 45
    },
    "🟡 Medium Risk PR - Moderate Complexity": {
        "cbo": 5, "wmc": 25, "dit": 2, "rfc": 30, "lcom": 80,
        "totalMethods": 12, "totalFields": 5, "nosi": 10, "loc": 150,
        "returnQty": 8, "loopQty": 5, "comparisonsQty": 18, "tryCatchQty": 2,
        "parenthesizedExpsQty": 8, "stringLiteralsQty": 15, "numbersQty": 20,
        "assignmentsQty": 25, "mathOperationsQty": 6, "variablesQty": 15,
        "maxNestedBlocks": 3, "uniqueWordsQty": 120
    },
    "🔴 High Risk PR - Complex Code": {
        "cbo": 12, "wmc": 65, "dit": 4, "rfc": 55, "lcom": 250,
        "totalMethods": 25, "totalFields": 8, "nosi": 30, "loc": 380,
        "returnQty": 18, "loopQty": 12, "comparisonsQty": 45, "tryCatchQty": 6,
        "parenthesizedExpsQty": 22, "stringLiteralsQty": 35, "numbersQty": 50,
        "assignmentsQty": 60, "mathOperationsQty": 15, "variablesQty": 35,
        "maxNestedBlocks": 5, "uniqueWordsQty": 280
    }
}

def get_risk_level(probability):
    """Get risk level from probability"""
    if probability >= 0.7:
        return "HIGH", "🔴"
    elif probability >= 0.4:
        return "MEDIUM", "🟡"
    else:
        return "LOW", "🟢"

def get_decision_path(model, X, feature_names):
    """Extract decision path"""
    tree = model.tree_
    node_indicator = model.decision_path(X)
    leaf_id = model.apply(X)
    
    sample_id = 0
    node_index = node_indicator.indices[
        node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]
    ]
    
    path_rules = []
    for node_id in node_index:
        if leaf_id[sample_id] == node_id:
            continue
        
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feature_name = feature_names[feature_idx]
        feature_value = X.iloc[sample_id, feature_idx]
        
        if feature_value <= threshold:
            comparison = "≤"
        else:
            comparison = ">"
        
        path_rules.append({
            'feature': feature_name,
            'comparison': comparison,
            'threshold': threshold,
            'value': feature_value
        })
    
    return path_rules

def predict_and_explain(model, features_dict):
    """Make prediction and generate explanation"""
    # Convert to DataFrame
    features_df = pd.DataFrame([features_dict])[FEATURE_COLUMNS]
    
    # Predict
    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0]
    defect_prob = probability[1]
    
    # Get risk level
    risk_level, risk_emoji = get_risk_level(defect_prob)
    
    # Get decision path
    decision_path = get_decision_path(model, features_df, FEATURE_COLUMNS)
    
    # Get feature importances
    importances = model.feature_importances_
    top_features_idx = np.argsort(importances)[::-1][:3]
    top_features = [
        {
            'name': FEATURE_COLUMNS[i],
            'value': features_dict[FEATURE_COLUMNS[i]],
            'importance': importances[i]
        }
        for i in top_features_idx
    ]
    
    return {
        'prediction': prediction,
        'probability': defect_prob,
        'risk_level': risk_level,
        'risk_emoji': risk_emoji,
        'decision_path': decision_path,
        'top_features': top_features
    }

# Main app
def main():
    # Header
    st.title("CommitGuard: PR Defect Predictor")
    st.markdown("""
    Predict whether a Pull Request is defect-prone using interpretable Decision Trees.
    Built with **76% accuracy** and **full explainability**.
    """)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2 = st.tabs(["Try Examples", "Upload Your Data"])
    
    # Tab 1: Examples
    with tab1:
        st.markdown("### Select an Example PR")
        st.markdown("Click a button to see instant prediction:")
        
        col1, col2, col3 = st.columns(3)
        
        selected_example = None
        
        with col1:
            if st.button("🟢 Low Risk\nSimple Code", use_container_width=True):
                selected_example = "🟢 Low Risk PR - Simple Code"
        
        with col2:
            if st.button("🟡 Medium Risk\nModerate", use_container_width=True):
                selected_example = "🟡 Medium Risk PR - Moderate Complexity"
        
        with col3:
            if st.button("🔴 High Risk\nComplex Code", use_container_width=True):
                selected_example = "🔴 High Risk PR - Complex Code"
        
        # Show results if example selected
        if selected_example or 'current_example' in st.session_state:
            if selected_example:
                st.session_state.current_example = selected_example
            
            example_name = st.session_state.current_example
            features = EXAMPLES[example_name]
            
            # Make prediction
            result = predict_and_explain(model, features)
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            # Risk box
            risk_color = {
                "HIGH": "#ff4444",
                "MEDIUM": "#ffaa00", 
                "LOW": "#44ff44"
            }[result['risk_level']]
            
            st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; background-color: {risk_color}20; border: 2px solid {risk_color};'>
                <h2 style='margin: 0; color: {risk_color};'>
                    {result['risk_emoji']} {result['risk_level']} RISK
                </h2>
                <h3 style='margin: 10px 0 0 0;'>
                    {result['probability']:.1%} Defect Probability
                </h3>
                <p style='margin: 5px 0 0 0; font-size: 18px;'>
                    Risk Score: {int(result['probability'] * 100)}/100
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")
            
            # Prediction
            if result['prediction'] == 1:
                st.error("**This PR is likely DEFECT-PRONE**")
            else:
                st.success("**This PR appears CLEAN**")
            
            # Decision Path
            st.markdown("### Decision Path")
            st.markdown("*How the model reached this decision:*")
            
            if result['decision_path']:
                for i, step in enumerate(result['decision_path'], 1):
                    st.markdown(f"""
                    **{i}.** `{step['feature']}` {step['comparison']} {step['threshold']:.2f}  
                    *(actual: {step['value']:.2f})*
                    """)
            else:
                st.markdown("*Single leaf decision - no conditions*")
            
            # Top Features
            st.markdown("### Top Risk Factors")
            st.markdown("*Most important features for this prediction:*")
            
            for i, feat in enumerate(result['top_features'], 1):
                importance_pct = feat['importance'] * 100
                st.markdown(f"""
                **{i}. {feat['name']}:** {feat['value']:.1f}  
                *Feature importance: {importance_pct:.1f}%*
                """)
            
            # Recommendation
            st.markdown("### Recommendation")
            if result['risk_level'] == "HIGH":
                st.warning("""
                - Assign senior reviewer
                - Increase test coverage
                - Consider breaking into smaller PRs
                - Add extra code review time
                """)
            elif result['risk_level'] == "MEDIUM":
                st.info("""
                - Standard code review process
                - Ensure adequate test coverage
                - Monitor for issues after merge
                """)
            else:
                st.success("""
                - Fast-track approval eligible
                - Standard testing sufficient
                - Low priority for senior review
                """)
    
    # Tab 2: Upload
    with tab2:
        st.markdown("### Upload PR Metrics (JSON)")
        st.markdown("Upload a JSON file with 21 code metrics:")
        
        uploaded_file = st.file_uploader(
            "Choose a JSON file",
            type=['json'],
            help="File should contain all 21 metrics (cbo, wmc, dit, etc.)"
        )
        
        if uploaded_file is not None:
            try:
                features = json.load(uploaded_file)
                
                # Validate features
                missing = set(FEATURE_COLUMNS) - set(features.keys())
                if missing:
                    st.error(f"Missing features: {', '.join(missing)}")
                else:
                    st.success("Valid metrics file!")
                    
                    if st.button("Predict Risk", use_container_width=True):
                        result = predict_and_explain(model, features)
                        
                        # Same display as examples
                        st.markdown("---")
                        st.markdown("## Prediction Results")
                        
                        st.markdown(f"### {result['risk_emoji']} {result['risk_level']} RISK")
                        st.metric("Defect Probability", f"{result['probability']:.1%}")
                        st.metric("Risk Score", f"{int(result['probability'] * 100)}/100")
                        
                        if result['prediction'] == 1:
                            st.error("This PR is likely DEFECT-PRONE")
                        else:
                            st.success("This PR appears CLEAN")
                        
                        # Show path and features
                        with st.expander("View Decision Path"):
                            for i, step in enumerate(result['decision_path'], 1):
                                st.write(f"{i}. {step['feature']} {step['comparison']} {step['threshold']:.2f} (actual: {step['value']:.2f})")
                        
                        with st.expander("View Top Features"):
                            for feat in result['top_features']:
                                st.write(f"- {feat['name']}: {feat['value']:.1f} (importance: {feat['importance']*100:.1f}%)")
                        
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        # Show example format
        with st.expander("View Example JSON Format"):
            example_json = EXAMPLES["🟡 Medium Risk PR - Moderate Complexity"]
            st.json(example_json)
            st.download_button(
                "Download Example JSON",
                data=json.dumps(example_json, indent=2),
                file_name="example_pr_metrics.json",
                mime="application/json"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About:** Built with Decision Trees for interpretable defect prediction.  
    **Model:** 76% accuracy, 80% ROC-AUC, depth-4 tree  
    **Dataset:** GHPR (6,052 GitHub PRs)  
    
    [GitHub Repository](https://github.com/manasmaskar/CommitGuard-ML-Defect-Predictor-Deployment) • 
    [View Code](https://github.com/manasmaskar/CommitGuard-ML-Defect-Predictor-Deployment/blob/main/src/predict.py)
    """)

if __name__ == "__main__":
    main()

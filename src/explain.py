'''Explainability utilities'''

import numpy as np
from sklearn.tree import _tree
import config


def get_decision_path(model, X, feature_names):
    '''Extract human-readable decision path'''
    tree = model.tree_
    
    # Get the path
    node_indicator = model.decision_path(X)
    leaf_id = model.apply(X)
    
    sample_id = 0
    node_index = node_indicator.indices[
        node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]
    ]
    
    path_rules = []
    for node_id in node_index:
        # Skip leaf node
        if leaf_id[sample_id] == node_id:
            continue
        
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feature_name = feature_names[feature_idx]
        feature_value = X.iloc[sample_id, feature_idx]
        
        if feature_value <= threshold:
            comparison = "<="
        else:
            comparison = ">"
        
        path_rules.append(
            f"{feature_name} {comparison} {threshold:.2f} (actual: {feature_value:.2f})"
        )
    
    return path_rules


def get_risk_level(probability):
    '''Convert probability to risk level'''
    if probability >= config.RISK_THRESHOLDS['HIGH']:
        return "HIGH ⚠️"
    elif probability >= config.RISK_THRESHOLDS['MEDIUM']:
        return "MEDIUM ⚡"
    else:
        return "LOW"


def get_top_features(model, feature_names, feature_values, top_n=3):
    '''Get top contributing features'''
    importances = model.feature_importances_
    
    # Get top N by importance
    top_idx = np.argsort(importances)[::-1][:top_n]
    
    top_features = []
    for idx in top_idx:
        top_features.append({
            'name': feature_names[idx],
            'value': feature_values[idx],
            'importance': importances[idx]
        })
    
    return top_features

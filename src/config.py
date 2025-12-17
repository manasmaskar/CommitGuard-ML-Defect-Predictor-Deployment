FEATURE_COLUMNS = [
    'cbo', 'wmc', 'dit', 'rfc', 'lcom', 'totalMethods', 'totalFields', 
    'nosi', 'loc', 'returnQty', 'loopQty', 'comparisonsQty', 'tryCatchQty',
    'parenthesizedExpsQty', 'stringLiteralsQty', 'numbersQty', 
    'assignmentsQty', 'mathOperationsQty', 'variablesQty', 
    'maxNestedBlocks', 'uniqueWordsQty'
]

MODEL_PATH = 'models/decision_tree_model.pkl'
METADATA_PATH = 'models/model_metadata.json'

# Risk thresholds - UPDATED based on analysis
RISK_THRESHOLDS = {
    'HIGH': 0.5,      # Above 50% = high risk
    'MEDIUM': 0.3,    # 30-50% = medium risk (optimal threshold)
    'LOW': 0.0        # Below 30% = low risk
}

# This reduces false negatives from 127 to 83 (34% reduction)
OPTIMAL_THRESHOLD = 0.30
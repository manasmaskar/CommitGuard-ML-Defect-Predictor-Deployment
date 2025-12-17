***
# CommitGuard: Explainable AI for Shift-Left Quality Control

Predict defect-prone pull requests using interpretable decision trees on static code metrics.

***

## Problem

Engineering teams waste time reviewing low-risk PRs while missing high-risk defects. This system predicts PR risk at submission time to prioritize reviews and catch bugs earlier.

***

## Results

**Model:** Decision Tree (depth=4)  
**Test Accuracy:** 76.4% --> 75.3% (optimized for recall)  
**ROC-AUC:** 80.4%  
**Recall:** 79% --> 86.3% (+7.3%)  
**False Negatives:** 127 --> 83 (34% reduction)

**Key Insight:** Lowering threshold from 0.50 to 0.30 catches 34% more defects while maintaining full interpretability.

***

## Dataset

**GHPR (GitHub Pull Request) Dataset**
- 6,052 real GitHub PRs 
- 21 static code metrics (CK suite, cyclomatic complexity, LOC, etc.)

- Source: https://github.com/feiwww/GHPR_dataset

***

## Features

✅ **Interpretable** - Shows exact decision path for every prediction  
✅ **Fast** - <10ms inference time  
✅ **Explainable** - Top contributing features + importance scores  
✅ **Production-ready** - CLI tool, tests, clean architecture  
✅ **Optimized** - Threshold tuning reduces missed defects by 34%

***

## Quick Start

### Install
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
```

### Train Model
```bash
jupyter notebook notebooks/02_model_development.ipynb
```

### Predict
```bash
python src/predict.py --input test_pr.json
```

### Test
```bash
python tests/test_basics.py
```

***

## Example Output

```
PR QUALITY GATE PREDICTION
============================================================
Defect Probability: 78%
Risk Score: 78/100
Risk Level: HIGH

Decision Path:
  1. nosi > 0.50 (actual: 22.00)
  2. totalMethods <= 223.00 (actual: 18.00)
  3. returnQty > 91.00 (actual: 95.00)

Top Contributing Features:
  1. nosi: 22.00 (importance: 0.8118)
  2. assignmentsQty: 40.00 (importance: 0.0722)
  3. cbo: 8.00 (importance: 0.0423)
```

***

## Project Structure

```
├── data/               # Dataset (raw + processed)
├── notebooks/          # EDA, training, analysis, optimization
├── src/                # Config, pipeline, prediction, explainability
├── models/             # Trained model + metadata
├── reports/figures/    # 10+ visualizations
├── tests/              # Unit tests
└── test_pr.json        # Sample input
```

***

## Decision Tree Fundamentals

**What this project demonstrates:**
- Hyperparameter tuning (depth, min_samples, criterion)
- Cost-complexity pruning
- Gini vs Entropy splitting
- Class imbalance handling with weights
- Threshold optimization for precision-recall trade-off
- Overfitting detection (train-test gap analysis)
- Feature importance analysis
- Interpretability vs accuracy trade-off

***

## Optimization Journey

**Tested 3 strategies:**

1. **Class-weight optimization** - Penalize false negatives more
2. **Random Forest ensemble** - 80.9% ROC-AUC (+0.5%)
3. **Threshold tuning** - Same recall, keeps interpretability 

**Chose threshold tuning:** 0.5% ROC-AUC gain from Random Forest doesn't justify losing explainability. Threshold tuning gives 34% fewer false negatives while maintaining transparent decision paths.

***

## Limitations

- **False Negatives:** 6.9% of defects still slip through (83/1210)
- **Uncertain predictions:** 26 samples (40-60% probability) have 27% accuracy
- **Edge cases:** Struggles with extreme values and novel patterns

**Mitigation:**  
Use as triage tool, not gatekeeper. Combine with static analysis. Escalate uncertain predictions to humans.


***

## Tech Stack

Python -  scikit-learn -  pandas -  matplotlib -  seaborn -  Jupyter

***

## Phase 2 Roadmap

- FastAPI REST service
- GitHub Actions webhook integration
- Automated PR risk comments
- Monitoring & drift detection
- Docker + AWS Lambda deployment

***

## Key Takeaway

**76% accuracy with full interpretability for production systems where users need to trust and act on predictions.**

***

## Dataset and Related Work



This project uses the GHPR dataset created by Xu et al. 

```
Jiaxi Xu, Fei Wang, Jun Ai. "Defect Prediction With Semantics and Context Features of Codes 
Based on Graph Representation Learning." IEEE Transactions on Reliability, 2021.
```

Dataset source: https://github.com/feiwww/GHPR_dataset


***

**Built to demonstrate decision tree fundamentals in a production context.**

***
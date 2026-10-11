"""Adaptive Data Intelligence Platform (ADIP).

A modular, offline-first analytics and machine-learning platform:

- ingest      : safe CSV/XLSX loading with validation and safeguards
- profiling   : schema detection, data-quality issues, target candidates
- planner     : automatic analytical-task recommendations
- preprocess  : leakage-safe sklearn preprocessing pipelines
- supervised  : classification/regression with baselines and honest evaluation
- unsupervised: clustering, anomaly detection, PCA with degenerate-case guards
- discovery   : evidence-based findings engine
- reliability : explicit, rule-based reliability statuses
- explain     : feature importance, confusion/residual/cluster profiles
- experiments : reproducible experiment logging and comparison
- scenario    : what-if analysis on trained supervised models
- nlq         : deterministic natural-language question answering
- reporting   : CSV / Excel / PDF report export

All analysis is local and deterministic. No external APIs are required.
"""

__version__ = "0.1.0"

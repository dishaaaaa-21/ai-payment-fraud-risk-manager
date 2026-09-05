\# AI Payment Fraud Risk Manager



An AI-powered payment fraud detection and risk management system combining machine learning, behavioral analysis, deterministic policy rules, explainable risk decisions, AI-assisted investigation, and human-in-the-loop review.



\## Overview



The system evaluates financial transactions through a multi-layer risk pipeline:



Transaction

→ Feature Engineering

→ ML Fraud Scoring

→ Behavioral Analysis

→ Policy Engine

→ Risk Fusion

→ AI Investigation

→ Case Management

→ Human Review

→ Audit Trail



The project is designed as an end-to-end fraud risk management MVP rather than only a machine learning classifier.



\## Key Features



\- XGBoost-based fraud detection model

\- Logistic Regression baseline comparison

\- Leakage-safe production feature pipeline

\- Time-based train, validation, and test split

\- PR-AUC focused evaluation for highly imbalanced fraud data

\- Behavioral risk detection

\- Transaction velocity analysis

\- Amount escalation detection

\- Counterparty repetition analysis

\- Deterministic policy engine

\- Multi-factor risk decision fusion

\- Dynamic entity blocklisting

\- AI-assisted fraud investigation

\- Human-in-the-loop case review

\- Full audit trail

\- FastAPI backend

\- Professional fintech-style frontend

\- REST API with 7 verified endpoints



\## Machine Learning Results



\### Selected Production Model



\*\*XGBoost (Production) with calibrated probabilities\*\*



Validation Results:



| Metric | Score |

|---|---:|

| PR-AUC | 0.9915 |

| ROC-AUC | 0.9999 |

| Precision | 0.9611 |

| Recall | 0.9334 |

| F1 Score | 0.9471 |



The test set remains untouched during architecture and development.



\## Risk Intelligence Pipeline



The system combines multiple sources of risk:



\### 1. Machine Learning Score



XGBoost predicts the probability of transaction fraud based on transaction-level features.



\### 2. Behavioral Engine



Tracks behavioral patterns including:



\- Transaction velocity

\- Amount escalation

\- Counterparty repetition

\- Deviation from historical behavior



\### 3. Policy Engine



Applies deterministic rules including:



\- `VELOCITY\_THRESHOLD`

\- `REPEATED\_HIGH\_VALUE`

\- `ABNORMAL\_ESCALATION`

\- `BLOCKLISTED\_ENTITY`



\### 4. Risk Fusion



Combines ML predictions, behavioral signals, and policy rules into a final risk tier:



\- LOW

\- MEDIUM

\- HIGH

\- CRITICAL



\## Synthetic Scenario



A synthetic transaction sequence demonstrates natural risk escalation:



| Amount | Risk Tier | Key Signal |

|---:|---|---|

| ₹1,200 | LOW | Clean transaction |

| ₹2,800 | LOW | Repeat counterparty, low amount |

| ₹8,000 | MEDIUM | Abnormal amount escalation |

| ₹20,000 | HIGH | Repeated high-value activity |

| ₹50,000 | CRITICAL | Velocity and multiple risk rules |



The entity starts clean and is dynamically blocklisted after behavioral detection and investigation.



\## Architecture



```text

Transaction

&#x20;   ↓

Feature Engineering

&#x20;   ↓

ML Fraud Model

&#x20;   ↓

Behavioral Engine

&#x20;   ↓

Policy Engine

&#x20;   ↓

Risk Decision Fusion

&#x20;   ↓

AI Investigation

&#x20;   ↓

Case Store

&#x20;   ↓

Human Review

&#x20;   ↓

Audit Trail



Project Structure

ai-payment-fraud-risk-manager/

│

├── dataset/                    # Dataset and acquisition scripts

├── models/                     # Trained ML models and metrics

├── src/

│   ├── api/                    # FastAPI backend

│   ├── behavioral.py           # Behavioral risk analysis

│   ├── policy.py               # Deterministic fraud rules

│   ├── risk\_decision.py        # Multi-signal risk fusion

│   ├── investigation.py        # AI-assisted investigation

│   ├── case\_store.py           # Fraud case management

│   └── audit.py                # Audit trail

│

├── frontend/

│   └── templates/              # Web interface

│

├── tests/                      # API and pipeline verification

├── run\_pipeline.py             # End-to-end pipeline runner

├── requirements.txt

├── NOTES.md

└── README.md


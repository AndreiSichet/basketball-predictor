# Basketball Predictor

A full-stack application that predicts NBA game outcomes using a self-trained machine learning model. Built as an end-to-end portfolio project covering the full pipeline: data ingestion, feature engineering, cloud-based model training and tracking, a Java backend serving predictions, and a React dashboard for exploring them.

## Tech stack

| Layer | Technology |
|---|---|
| Data pipeline & preprocessing | Python (pandas, requests, nba_api) |
| Model training & experimentation | Python (scikit-learn, XGBoost/LightGBM), MLflow, cloud training (SageMaker / Vertex AI) |
| Model serving | Containerized inference API (FastAPI), Docker |
| Backend | Java, Spring Boot, PostgreSQL |
| Frontend | React |
| Infrastructure & CI/CD | Docker, GitHub Actions, Terraform |

## Project structure

basketball-predictor/
├── data-pipeline/ # Ingestion and preprocessing (Python)
│ ├── ingestion/ # Scripts pulling raw game and player data
│ ├── preprocessing/ # Cleaning and feature engineering
│ ├── data/
│ │ ├── raw/ # Untouched pulls (gitignored)
│ │ └── processed/ # Engineered features (gitignored)
│ └── requirements.txt
├── ml-training/ # Model development and experiments (Python)
│ ├── notebooks/ # Exploratory data analysis
│ ├── models/ # Saved model artifacts (gitignored)
│ └── mlruns/ # MLflow experiment tracking (gitignored)
├── inference-service/ # Containerized model-serving API
├── backend/ # Java Spring Boot REST API + PostgreSQL
├── frontend/ # React dashboard
├── infra/ # Terraform and deployment configuration
├── .github/workflows/ # CI/CD pipelines
├── basketball-predictor.code-workspace # VS Code multi-root workspace
└── README.md


## Status

This project is under active development.

- [x] Repository structure and multi-root VS Code workspace
- [x] Spring Boot backend skeleton (builds successfully)
- [x] React frontend skeleton
- [ ] Data pipeline: NBA game and player data ingestion
- [ ] Feature engineering (team ratings, rest days, home/away splits, pace-adjusted efficiency)
- [ ] Model training and experiment tracking in the cloud
- [ ] MLOps: model registry, automated retraining, CI/CD
- [ ] Backend API and database integration
- [ ] Frontend dashboard
- [ ] Cloud deployment
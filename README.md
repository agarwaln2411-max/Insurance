
Insurance / HR Attrition Streamlit Dashboard
===========================================

Files at repository root (no folders):
- app.py            -> main Streamlit app
- Insurance.csv     -> sample dataset included
- requirements.txt  -> package list (no pinned versions)
- README.md         -> this file

Quick deploy:
1. Push all files in this ZIP to a GitHub repo (root).
2. Connect the repo to Streamlit Cloud (share.streamlit.io).
3. Set the main file as app.py and deploy.

Dashboard features:
- 5 actionable charts for retention insights (attrition by role, satisfaction vs tenure, compensation analysis, cohort/heatmap or correlation, feature importance)
- Global filters: Job Role multi-select + Satisfaction slider
- Train & Evaluate tab: Decision Tree, Random Forest, Gradient Boosting with metrics and plots
- Predict tab: Upload new dataset, predict attrition, download predictions.csv
- Uses automatic column detection; edit app.py if you want to force specific column names.

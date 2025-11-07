
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve)
import plotly.express as px
import base64

st.set_page_config(page_title="HR / Insurance Attrition Dashboard", layout="wide")

@st.cache_data
def load_file(f):
    try:
        return pd.read_excel(f)
    except Exception:
        f.seek(0)
        return pd.read_csv(f)

def detect_columns(df):
    cols = {'jobrole': None, 'satisfaction': None, 'label': None}
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ['jobrole','job role','role','position']):
            cols['jobrole'] = c
        if any(k in lc for k in ['satisfaction','satis','engagement','job satisfaction','satisfaction_level','satisfaction_score','satis_score']):
            cols['satisfaction'] = c
        if any(k in lc for k in ['policy_status','attrition','left','resigned','target','label']):
            cols['label'] = c
    # fallback heuristics
    if cols['jobrole'] is None:
        for c in df.select_dtypes(include=['object','category']).columns:
            if df[c].nunique() < 50:
                cols['jobrole'] = c
                break
    if cols['satisfaction'] is None:
        for c in df.select_dtypes(include=[np.number]).columns:
            if 'score' in c.lower() or 'rating' in c.lower() or 'satis' in c.lower():
                cols['satisfaction'] = c
                break
    if cols['label'] is None:
        last = df.columns[-1]
        if df[last].dropna().nunique() <= 5:
            cols['label'] = last
    return cols

def preprocess_for_model(df, label_col):
    df = df.copy()
    df = df.loc[:, df.isnull().mean() < 0.6]
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes(include=['object','category']).columns:
        df[c] = df[c].fillna('Unknown')
    encoders = {}
    for c in df.select_dtypes(include=['object','category']).columns:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y, encoders

# Sidebar
st.sidebar.title("Data & Filters")
uploaded = st.sidebar.file_uploader("Upload dataset (CSV/XLSX). If none, sample Insurance.csv included.", type=['csv','xlsx'])
if uploaded is None:
    try:
        df = pd.read_csv("Insurance.csv")
    except Exception:
        st.sidebar.error("No dataset found. Please upload Insurance.csv or another dataset.")
        st.stop()
else:
    df = load_file(uploaded)

st.sidebar.write("Dataset shape:", df.shape)
cols_detected = detect_columns(df)
st.sidebar.write("Auto-detected columns:", cols_detected)

job_col = cols_detected.get('jobrole')
satis_col = cols_detected.get('satisfaction')
label_col = cols_detected.get('label')

st.title("HR / Insurance Attrition Dashboard")
st.markdown("Interactive dashboard for insights and quick ML predictions. Upload your own dataset or use the included Insurance.csv.")

# Filters in sidebar
st.sidebar.markdown("### Filters")
if job_col and job_col in df.columns:
    job_options = sorted(df[job_col].dropna().unique().tolist())
    selected_jobs = st.sidebar.multiselect("Job Role (multi-select)", options=job_options, default=job_options)
else:
    selected_jobs = None

if satis_col and satis_col in df.columns:
    min_s, max_s = float(df[satis_col].min()), float(df[satis_col].max())
    s_low, s_high = st.sidebar.slider("Satisfaction range", min_value=min_s, max_value=max_s, value=(min_s, max_s))
else:
    s_low, s_high = None, None

# Apply filters
df_filt = df.copy()
if job_col and selected_jobs is not None:
    df_filt = df_filt[df_filt[job_col].isin(selected_jobs)]
if satis_col and s_low is not None:
    df_filt = df_filt[(df_filt[satis_col] >= s_low) & (df_filt[satis_col] <= s_high)]

# Tabs
tabs = st.tabs(["Overview & Charts", "Train & Evaluate Models", "Predict New Data", "Data Preview & Export"])

with tabs[0]:
    st.header("Five Key Charts & Complex Insights")
    # Chart 1: Attrition rate by Job Role (stacked bar showing proportions)
    st.subheader("1) Attrition / Policy Status Proportion by Job Role")
    if label_col and job_col:
        prop = df_filt.groupby(job_col)[label_col].value_counts(normalize=True).unstack(fill_value=0)
        fig = px.bar(prop, barmode='stack', title="Proportion of labels by Job Role")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("- Insight: roles with high proportion of 'left' (or negative label) are priority for retention programs.")
    else:
        st.info("Job role or label not detected for Chart 1.")

    # Chart 2: Satisfaction vs Tenure (scatter with trendline) - detect tenure-like columns
    st.subheader("2) Satisfaction vs Tenure / Years in Company (Scatter + Trendline)")
    tenure_cols = [c for c in df_filt.columns if any(k in c.lower() for k in ['tenure','year','service','experience','yrs','years'])]
    tenure_col = tenure_cols[0] if tenure_cols else None
    if satis_col:
        if tenure_col:
            fig2 = px.scatter(df_filt, x=tenure_col, y=satis_col, color=label_col if label_col in df_filt.columns else None, trendline="ols",
                              title=f"Satisfaction vs {tenure_col} (color=label)")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("- Insight: long-tenure employees with low satisfaction are high-risk retention targets.")
        else:
            fig2 = px.histogram(df_filt, x=satis_col, nbins=20, title="Satisfaction distribution")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No satisfaction-like column detected for Chart 2.")

    # Chart 3: Compensation vs Attrition (box plots)
    st.subheader("3) Compensation / Premium / Income distribution by Label (Box Plot)")
    money_cols = [c for c in df_filt.columns if any(k in c.lower() for k in ['salary','income','pay','compensation','monthly','premium','amt'])]
    money_col = money_cols[0] if money_cols else None
    if money_col and label_col:
        fig3 = px.box(df_filt, x=label_col, y=money_col, points="outliers", title=f"{money_col} by {label_col}")
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("- Insight: significant pay differences between stayed vs left may indicate pay-based churn.")
    else:
        st.info("No compensation-like column or label detected for Chart 3.")

    # Chart 4: Cohort retention-like chart (group by hire year or join year if available)
    st.subheader("4) Cohort / Join Year retention overview (heatmap-styled)")
    join_cols = [c for c in df_filt.columns if any(k in c.lower() for k in ['join','hire','start','date'])]
    join_col = join_cols[0] if join_cols else None
    if join_col and label_col:
        try:
            df_temp = df_filt.copy()
            df_temp['join_year'] = pd.to_datetime(df_temp[join_col], errors='coerce').dt.year.fillna(0).astype(int)
            cohort = df_temp.groupby(['join_year', label_col]).size().unstack(fill_value=0)
            cohort_prop = cohort.div(cohort.sum(axis=1), axis=0)
            fig4 = px.imshow(cohort_prop.fillna(0), text_auto=True, aspect='auto', title="Join Year vs Label proportion")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("- Insight: cohorts with high early attrition need onboarding review.")
        except Exception:
            st.info("Could not build cohort chart from join/hire column.")
    else:
        st.info("No join/hire date column detected for Chart 4. Showing correlation heatmap instead.")
        num = df_filt.select_dtypes(include=[np.number])
        if not num.empty:
            corr = num.corr()
            fig4b = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation matrix (numeric features)")
            st.plotly_chart(fig4b, use_container_width=True)

    # Chart 5: Feature importance from Quick RandomForest (if label detected)
    st.subheader("5) Quick Feature Importance (RandomForest)")
    if label_col:
        try:
            df_model = df_filt.dropna(subset=[label_col]).copy()
            if df_model.shape[0] > 30:
                Xq, yq, _ = preprocess_for_model(df_model, label_col)
                rf = RandomForestClassifier(n_estimators=200, random_state=42)
                rf.fit(Xq, yq)
                imp = pd.Series(rf.feature_importances_, index=Xq.columns).sort_values(ascending=False).head(20)
                fig5 = px.bar(imp, title="Top feature importances (RandomForest)")
                st.plotly_chart(fig5, use_container_width=True)
                st.markdown("- Insight: features above are strong predictors of attrition and can guide HR actions.")
            else:
                st.info("Not enough rows to compute feature importances reliably (min 30 rows).")
        except Exception as e:
            st.error(f"Feature importance error: {e}")
    else:
        st.info("Label column not detected; cannot compute feature importance.")

with tabs[1]:
    st.header("Train & Evaluate Models (DecisionTree, RandomForest, GradientBoosting)")
    st.write("Click the button below to train all three models on the filtered dataset and compute metrics.")
    if label_col is None:
        st.error("No label column detected — cannot train models. Ensure dataset has an attrition/policy_status label column.")
    else:
        run = st.button("Train & Evaluate All Models")
        if run:
            with st.spinner("Training models..."):
                try:
                    df_train = df_filt.dropna(subset=[label_col]).copy()
                    X, y, encs = preprocess_for_model(df_train, label_col)
                    if X.shape[0] < 30:
                        st.warning("Training data is small (<30 rows). Results may be unreliable.")
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
                    models = {
                        "Decision Tree": DecisionTreeClassifier(random_state=42),
                        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
                        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
                    }
                    results = {}
                    for name, mdl in models.items():
                        mdl.fit(X_train, y_train)
                        y_pred = mdl.predict(X_test)
                        y_proba = mdl.predict_proba(X_test)[:,1] if hasattr(mdl, "predict_proba") else None
                        cm = confusion_matrix(y_test, y_pred)
                        acc = accuracy_score(y_test, y_pred)
                        prec = precision_score(y_test, y_pred, zero_division=0)
                        rec = recall_score(y_test, y_pred, zero_division=0)
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                        auc = roc_auc_score(y_test, y_proba) if y_proba is not None and len(np.unique(y_test))==2 else None
                        cv = cross_val_score(mdl, X, y, cv=5, scoring='accuracy')
                        fi = pd.Series(mdl.feature_importances_, index=X.columns).sort_values(ascending=False).head(30) if hasattr(mdl, "feature_importances_") else None
                        results[name] = {"model": mdl, "cm": cm, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc, "cv_mean": cv.mean(), "cv_std": cv.std(), "feature_importance": fi}
                    st.session_state['trained_models'] = results
                    # Display results
                    for name, r in results.items():
                        st.subheader(name)
                        st.write(f"Accuracy: {r['accuracy']:.3f} | Precision: {r['precision']:.3f} | Recall: {r['recall']:.3f} | F1: {r['f1']:.3f}")
                        if r['auc'] is not None:
                            st.write(f"AUC: {r['auc']:.3f}")
                        st.write(f"Cross-val accuracy: {r['cv_mean']:.3f} ± {r['cv_std']:.3f}")
                        st.write("Confusion Matrix:")
                        cm_df = pd.DataFrame(r['cm'], index=['Actual 0','Actual 1'], columns=['Pred 0','Pred 1'])
                        st.dataframe(cm_df)
                        if r['feature_importance'] is not None:
                            st.plotly_chart(px.bar(r['feature_importance'], title=f"{name} - Feature importance"), use_container_width=True)
                    st.success("Training complete. Models stored in session for use in prediction tab.")
                except Exception as e:
                    st.error(f"Training failed: {e}")

with tabs[2]:
    st.header("Upload New Dataset & Predict Attrition")
    st.write("Upload a new dataset (CSV/XLSX). The app will align columns where possible, predict attrition using the trained Random Forest (or will train a quick RF if none), and allow download of predictions.")
    new_file = st.file_uploader("Upload dataset for prediction", key="new_upload")
    if new_file is not None:
        new_df = load_file(new_file)
        st.write("Preview of uploaded data:")
        st.dataframe(new_df.head())
        if st.button("Run Predictions on uploaded file"):
            try:
                # choose model
                model = None
                if 'trained_models' in st.session_state and "Random Forest" in st.session_state['trained_models']:
                    model = st.session_state['trained_models']["Random Forest"]['model']
                if model is None:
                    st.info("No trained Random Forest in session — training a quick RF on the original dataset.")
                    base = df.dropna(subset=[label_col]).copy()
                    Xb, yb, _ = preprocess_for_model(base, label_col)
                    model = RandomForestClassifier(n_estimators=200, random_state=42)
                    model.fit(Xb, yb)
                # Align columns: use training numeric columns intersection as features
                base = df.copy()
                combined = pd.concat([base, new_df], sort=False, ignore_index=True)
                for c in combined.select_dtypes(include=[np.number]).columns:
                    combined[c] = combined[c].fillna(combined[c].median())
                for c in combined.select_dtypes(include=['object','category']).columns:
                    combined[c] = combined[c].fillna('Unknown')
                for c in combined.select_dtypes(include=['object','category']).columns:
                    le = LabelEncoder()
                    combined[c] = le.fit_transform(combined[c].astype(str))
                new_prepped = combined.iloc[len(base):].copy()
                train_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else Xb.columns.tolist()
                intersect = [c for c in train_features if c in new_prepped.columns]
                if not intersect:
                    st.error("Unable to align features between training data and uploaded file. Ensure compatible schema.")
                else:
                    Xpred = new_prepped[intersect].fillna(0)
                    preds = model.predict(Xpred)
                    out = new_df.copy()
                    out['predicted_label'] = preds
                    st.write("Predictions (first 10 rows):")
                    st.dataframe(out.head(10))
                    csv = out.to_csv(index=False).encode()
                    b64 = base64.b64encode(csv).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download predictions.csv</a>'
                    st.markdown(href, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with tabs[3]:
    st.header("Data Preview & Export")
    st.write("Preview the filtered dataset and download for reporting.")
    st.dataframe(df_filt.head(200))
    if st.button("Download filtered dataset as CSV"):
        csv = df_filt.to_csv(index=False).encode()
        st.download_button("Download CSV", data=csv, file_name="filtered_dataset.csv", mime="text/csv")

st.sidebar.markdown("---")
st.sidebar.markdown("Notes: app auto-detects Job Role, Satisfaction, and Label columns where possible. Edit column names in code if detection fails.")

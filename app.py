import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Metacognitive XAI Dashboard", layout="wide")

st.title("📊 Academic Performance XAI Dashboard")
st.markdown("### Enhancing Student Metacognition Through Explainable AI")

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    model_path = "best_model.pkl"
    if not os.path.exists(model_path):
        st.error("Model file not found.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Upload Student Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

# =====================================================
# MAIN LOGIC
# =====================================================

if uploaded_file is not None:

    try:
        # -----------------------------
        # READ FILE
        # -----------------------------
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("📂 Dataset Preview")
        st.dataframe(df.head())

        # -----------------------------
        # BASIC CLEANING
        # -----------------------------
        if "id_student" in df.columns:
            df = df.drop(columns=["id_student"])

        if "date_submitted" in df.columns and "assessment_deadline" in df.columns:
            df["date_submitted"] = pd.to_datetime(df["date_submitted"], errors="coerce")
            df["assessment_deadline"] = pd.to_datetime(df["assessment_deadline"], errors="coerce")
            df["days_diff"] = (df["assessment_deadline"] - df["date_submitted"]).dt.days

        required_features = [
            "Sum of sum_click",
            "Sum of late_flag",
            "total_submissions",
            "late_%",
            "studied_credits",
            "age_band",
            "gender",
            "disability",
            "days_diff"
        ]

        df_model = df[required_features].copy()
        df_model = df_model.replace([np.inf, -np.inf], np.nan)
        df_model = df_model.fillna(0)

        # -----------------------------
        # RUN PREDICTION
        # -----------------------------
        if st.sidebar.button("Run Prediction"):
            predictions = model.predict(df_model)

            st.session_state["predictions"] = predictions
            st.session_state["df_model"] = df_model
            st.session_state["df_full"] = df

        # =====================================================
        # DISPLAY RESULTS (PERSISTED)
        # =====================================================

        if "predictions" in st.session_state:

            predictions = st.session_state["predictions"]
            df_model = st.session_state["df_model"]
            df = st.session_state["df_full"]

            result_df = df.copy()
            result_df["Predicted Score"] = predictions

            st.subheader("🔮 Predicted Scores")
            st.dataframe(result_df)

            # =====================================================
            # MODEL PERFORMANCE + CONFIDENCE INTERVAL
            # =====================================================

            if "Average of score" in df.columns:

                actual = df["Average of score"]
                mask = ~actual.isna()

                actual_clean = actual[mask]
                pred_clean = predictions[mask]

                mse = mean_squared_error(actual_clean, pred_clean)
                rmse = np.sqrt(mse)
                r2 = r2_score(actual_clean, pred_clean)

                st.subheader("📈 Model Performance")

                col1, col2, col3 = st.columns(3)
                col1.metric("MSE", round(mse, 3))
                col2.metric("RMSE", round(rmse, 3))
                col3.metric("R²", round(r2, 3))

                st.markdown("**Prediction Confidence Interval:** Predicted Score ± RMSE")

                # Residual plot
                residuals = actual_clean - pred_clean
                fig_res, ax_res = plt.subplots()
                ax_res.scatter(pred_clean, residuals, alpha=0.4)
                ax_res.axhline(0, linestyle="--")
                ax_res.set_xlabel("Predicted Score")
                ax_res.set_ylabel("Residual")
                ax_res.set_title("Residual Analysis")
                st.pyplot(fig_res)

            # =====================================================
            # RISK DISTRIBUTION
            # =====================================================

            st.subheader("🎯 Risk Distribution")

            def risk_label(score):
                if score < 60:
                    return "High Risk"
                elif score < 75:
                    return "Moderate Risk"
                else:
                    return "Low Risk"

            risk = pd.Series(predictions).apply(risk_label)
            risk_counts = risk.value_counts()

            fig_risk, ax_risk = plt.subplots()
            ax_risk.bar(risk_counts.index, risk_counts.values)
            ax_risk.set_title("Student Risk Categories")
            st.pyplot(fig_risk)

            # =====================================================
            # FAIRNESS ANALYSIS
            # =====================================================

            st.subheader("⚖️ Fairness Analysis")

            if "gender" in df.columns:
                gender_avg = pd.DataFrame({
                    "gender": df["gender"],
                    "prediction": predictions
                }).groupby("gender")["prediction"].mean()

                fig_g, ax_g = plt.subplots()
                ax_g.bar(gender_avg.index, gender_avg.values)
                ax_g.set_title("Average Prediction by Gender")
                st.pyplot(fig_g)

            # =====================================================
            # SHAP GLOBAL + INDIVIDUAL
            # =====================================================

            st.subheader("🌍 Global Feature Importance")

            preprocessor = model.named_steps["preprocessor"]
            reg_model = model.named_steps["model"]

            transformed = preprocessor.transform(df_model)
            raw_names = preprocessor.get_feature_names_out()

            clean_names = [
                name.replace("num__", "")
                    .replace("cat__", "")
                    .replace("_", " ")
                for name in raw_names
            ]

            transformed_df = pd.DataFrame(transformed, columns=clean_names)

            explainer = shap.Explainer(reg_model, transformed_df)
            shap_values = explainer(transformed_df)

            mean_abs = np.abs(shap_values.values).mean(axis=0)

            shap_df = pd.DataFrame({
                "Feature": clean_names,
                "Importance": mean_abs
            }).sort_values(by="Importance", ascending=True)

            fig_shap, ax_shap = plt.subplots(figsize=(8, 6))
            ax_shap.barh(shap_df["Feature"], shap_df["Importance"])
            ax_shap.set_title("Global SHAP Importance")
            st.pyplot(fig_shap)

            # =====================================================
            # INDIVIDUAL ANALYSIS
            # =====================================================

            st.subheader("👤 Individual Student Analysis")

            selected = st.selectbox(
                "Select Student Index",
                range(len(transformed_df))
            )

            individual_values = shap_values.values[selected]

            individual_df = pd.DataFrame({
                "Feature": clean_names,
                "SHAP Value": individual_values
            }).sort_values(by="SHAP Value")

            fig_ind, ax_ind = plt.subplots(figsize=(8, 6))
            colors = ["#A3C4DC" if v < 0 else "#6C8EBF"
                      for v in individual_df["SHAP Value"]]

            ax_ind.barh(
                individual_df["Feature"],
                individual_df["SHAP Value"],
                color=colors
            )

            ax_ind.set_title("Individual Feature Impact")
            st.pyplot(fig_ind)

            # =====================================================
            # REFLECTION PROMPTS
            # =====================================================

            st.subheader("🧠 Personalized Reflection")

            negative = individual_df[individual_df["SHAP Value"] < 0].head(3)

            for _, row in negative.iterrows():
                feature = row["Feature"]

                if "late" in feature.lower():
                    prompt = "Your submission timing appears to influence performance. How can you improve deadline management?"
                elif "click" in feature.lower():
                    prompt = "Engagement level impacts your performance. Are you actively interacting with learning materials?"
                elif "studied credits" in feature.lower():
                    prompt = "Credit load may affect performance. Is workload balance optimized?"
                else:
                    prompt = "This factor influences your predicted score. What strategy could improve it?"

                st.markdown(f"🔍 **Reflection Area: {feature}**")
                st.write(prompt)

            # =====================================================
            # SELF-ASSESSMENT WITH FEEDBACK
            # =====================================================

            st.subheader("📝 Metacognitive Self-Assessment")

            prep = st.slider("Preparedness Level", 1, 5)
            deadline = st.slider("Deadline Management", 1, 5)
            strategy = st.slider("Study Strategy Consistency", 1, 5)

            meta_score = (prep + deadline + strategy) / 3

            if meta_score < 2.5:
                st.warning("⚠ Strategic alignment appears low. Consider structured improvement plans.")
            elif meta_score < 4:
                st.info("ℹ Moderate strategic awareness detected.")
            else:
                st.success("✅ Strong metacognitive alignment detected.")

            # =====================================================
            # WHAT-IF SIMULATOR
            # =====================================================

            st.subheader("🔄 What-If Simulation")

            selected_feature = st.selectbox(
                "Select Feature to Modify",
                df_model.columns
            )

            current_value = df_model.iloc[selected][selected_feature]

            new_value = st.slider(
                f"Adjust {selected_feature}",
                float(df_model[selected_feature].min()),
                float(df_model[selected_feature].max()),
                float(current_value)
            )

            modified_instance = df_model.iloc[[selected]].copy()
            modified_instance[selected_feature] = new_value

            original_pred = model.predict(df_model.iloc[[selected]])[0]
            new_pred = model.predict(modified_instance)[0]

            col1, col2 = st.columns(2)
            col1.metric("Original Score", round(original_pred, 2))
            col2.metric("New Score", round(new_pred, 2),
                        delta=round(new_pred - original_pred, 2))

            # =====================================================
            # LEARNING TRAJECTORY
            # =====================================================

            st.subheader("📈 Learning Trajectory Simulation")

            trajectory_values = np.linspace(
                df_model[selected_feature].min(),
                df_model[selected_feature].max(),
                10
            )

            simulated_scores = []

            for val in trajectory_values:
                temp = df_model.iloc[[selected]].copy()
                temp[selected_feature] = val
                simulated_scores.append(model.predict(temp)[0])

            fig_traj, ax_traj = plt.subplots()
            ax_traj.plot(trajectory_values, simulated_scores)
            ax_traj.set_xlabel(selected_feature)
            ax_traj.set_ylabel("Predicted Score")
            ax_traj.set_title("Projected Learning Path")
            st.pyplot(fig_traj)

    except Exception as e:
        st.error("Error occurred.")
        st.write(str(e))

else:
    st.info("Upload a dataset to begin.")
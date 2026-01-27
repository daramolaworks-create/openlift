import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from pathlib import Path
from openlift.core.pipeline import run_geo_lift_df

st.set_page_config(page_title="OpenLift UI", layout="wide")

st.title("OpenLift: Geo-Lift Experiment Runner")

# 1. Upload Data
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (date, geo, outcome)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Display raw data snapshot
        with st.expander("Data Preview"):
            st.dataframe(df.head())
        
        # 2. Configuration
        st.sidebar.header("2. Configuration")
        
        # Column Mapping
        st.sidebar.subheader("Column Mapping")
        date_col = st.sidebar.selectbox("Date Column", df.columns, index=0)
        geo_col = st.sidebar.selectbox("Geo Column", df.columns, index=1)
        outcome_col = st.sidebar.selectbox("Outcome Metric (Y)", df.columns, index=2, help="The goal metric you want to increase (e.g. Purchases, Revenue)")
        
        # Cost/Input Metric
        cost_col = st.sidebar.selectbox("Input Metric (X) [Optional]", ["None"] + list(df.columns), index=0, help="The metric you changed/increased (e.g. Amount Spend, Impressions). Used to calculate ROI.")
        
        # Pre-processing dates
        df[date_col] = pd.to_datetime(df[date_col])
        min_date = df[date_col].min().date()
        max_date = df[date_col].max().date()
        
        st.sidebar.info(f"📅 Entire Data Range: **{min_date}** to **{max_date}**")
        
        # Geo Selection
        all_geos = df[geo_col].unique().tolist()
        test_geo = st.sidebar.selectbox("Test Geo (Where ads ran)", all_geos)
        control_geos = st.sidebar.multiselect("Control Geos", [g for g in all_geos if g != test_geo], default=[g for g in all_geos if g != test_geo])
        
        # Period Selection
        st.sidebar.subheader("3. Define Periods")
        st.sidebar.markdown("The **Pre-Period** trains the model. The **Post-Period** is where you measure impact.")
        
        pre_start = st.sidebar.date_input("Pre-Period Start", min_date, min_value=min_date, max_value=max_date)
        pre_end = st.sidebar.date_input("Pre-Period End", min_date, min_value=min_date, max_value=max_date)
        
        post_start = st.sidebar.date_input("Post-Period Start", max_date, min_value=min_date, max_value=max_date)
        post_end = st.sidebar.date_input("Post-Period End", max_date, min_value=min_date, max_value=max_date)
        
        # 3. Run Button
        if st.sidebar.button("🚀 Run Experiment", type="primary"):
            if not control_geos:
                st.error("Please select at least 2 control geos.")
            elif len(control_geos) < 2:
                st.error("Model requires at least 2 Control Geos to work correctly.")
            else:
                with st.spinner("Running Bayesian Model... isolating seasonality..."):
                    try:
                        results = run_geo_lift_df(
                            df=df,
                            test_geo=test_geo,
                            control_geos=control_geos,
                            pre_start=str(pre_start),
                            pre_end=str(pre_end),
                            post_start=str(post_start),
                            post_end=str(post_end),
                            date_col=date_col,
                            geo_col=geo_col,
                            outcome_col=outcome_col
                        )
                        
                        # --- ANALYZE INPUT (COST) CHANGE ---
                        input_change_pct = 0.0
                        input_change_abs = 0.0
                        has_input = cost_col != "None"
                        
                        if has_input:
                            # Filter for Test Geo
                            mask_geo = df[geo_col] == test_geo
                            
                            # Pre Period Stats
                            mask_pre = (df[date_col] >= pd.Timestamp(pre_start)) & (df[date_col] <= pd.Timestamp(pre_end))
                            pre_val = df.loc[mask_geo & mask_pre, cost_col].mean()
                            
                            # Post Period Stats
                            mask_post = (df[date_col] >= pd.Timestamp(post_start)) & (df[date_col] <= pd.Timestamp(post_end))
                            post_sum = df.loc[mask_geo & mask_post, cost_col].sum()
                            post_mean = df.loc[mask_geo & mask_post, cost_col].mean()
                            
                            # Calculate Lift in Input (Simple Pre-Post difference, naive)
                            # Better: Just show the % increase in daily average
                            if pre_val > 0:
                                input_change_pct = ((post_mean - pre_val) / pre_val) * 100
                            input_change_abs = post_sum - (pre_val * (df[mask_geo & mask_post].shape[0])) # Total Incremental Spend approx
                            
                        # --- PRE-COMPUTE DATA ---
                        m = results['metrics']
                        coeffs = m.get('model_coefficients', [])
                        dow = m.get('dow_effect', [])
                        
                        # 4. Results Section
                        st.balloons()
                        st.divider()
                        st.header("Results Analysis")
                        
                        # --- KPI Cards ---
                        inc_mean = m['incremental_outcome_mean']
                        lift_pct = m['lift_pct_mean']
                        prob_pos = m['p_positive']
                        obs_total = m.get('observed_outcome_sum', 0.0)
                        pred_base = m.get('predicted_outcome_mean', 0.0)
                        
                        cols = st.columns(4)
                        cols[0].metric(
                            "Incremental Lift (Y)", 
                            f"{inc_mean:+.1f}",
                            f"Range: {m['incremental_outcome_hdi_90'][0]:.1f} to {m['incremental_outcome_hdi_90'][1]:.1f}",
                            delta_color="normal"
                        )
                        cols[1].metric(
                            "Lift % (Y)", 
                            f"{lift_pct:+.1f}%",
                            f"Observed: {obs_total:.1f} vs Baseline: {pred_base:.1f}"
                        )
                        
                        prob_color = "green" if prob_pos > 0.9 else "orange"
                        cols[2].metric("Confidence", f"{prob_pos*100:.1f}%", "Significant" if prob_pos > 0.9 else "Low Confidence")

                        # Efficiency Metric
                        if has_input:
                            cols[3].metric(
                                f"Input Shift ({cost_col})", 
                                f"{input_change_pct:+.1f}%",
                                f"Est. Delta: {input_change_abs:+.1f}"
                            )
                        else:
                            cols[3].metric("Input Shift", "-", "Select Input Col")

                        # --- DETAILED INSIGHTS ---
                        st.divider()
                        st.header("💡 Strategic Insights")
                        
                        tab1, tab2, tab3 = st.tabs(["Conclusion", "Seasonality & Trends", "Model Weights"])
                        
                        with tab1:
                            st.subheader("Executive Summary")
                            
                            # Dynamic Narrative Generation based on 4 quadrants
                            narrative = ""
                            
                            # 1. Did Input change?
                            if has_input:
                                narrative += f"**1. The Action:** You changed `{cost_col}` in **{test_geo}** by **{input_change_pct:+.1f}%** during the test period.\n\n"
                            else:
                                narrative += f"**1. The Action:** We analyzed the performance of **{test_geo}** during the test period.\n\n"
                                
                            # 2. Did Outcome change?
                            if prob_pos > 0.9:
                                narrative += f"**2. The Result:** This drove a **statistically significant increase** in `{outcome_col}` of **{lift_pct:+.1f}%** (Confidence: {prob_pos*100:.1f}%).\n\n"
                            else:
                                narrative += f"**2. The Result:** We observed a small change in `{outcome_col}` ({lift_pct:+.1f}%), but it is **not statistically significant** (Confidence: {prob_pos*100:.1f}%). It is indistinguishable from random noise.\n\n"
                            
                            # 3. Efficiency / Elasticity
                            if has_input and inc_mean > 0 and input_change_abs > 0:
                                cpi = input_change_abs / inc_mean
                                narrative += f"**3. Efficiency:** You spent roughly **{input_change_abs:.1f}** {cost_col} to get **{inc_mean:.1f}** extra {outcome_col}.\n"
                                narrative += f"   *   **Cost Per Incremental Lift:** {cpi:.2f}\n"
                                
                                # Elasticity
                                if input_change_pct > 0:
                                    elasticity = lift_pct / input_change_pct
                                    narrative += f"   *   **Elasticity:** {elasticity:.2f} (For every 1% increase in Input, you got {elasticity:.2f}% increase in Outcome)."
                            
                            st.markdown(narrative)
                            
                        with tab2:
                            st.subheader("Did we account for Seasonality?")
                            st.write("Yes. The model learned the following 'Day of Week' patterns from your data:")
                            
                            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                            if len(dow) == 7:
                                dow_df = pd.DataFrame({"Day": days, "Effect": dow})
                                
                                c1, c2 = st.columns([1, 2])
                                with c2:
                                    st.bar_chart(dow_df.set_index("Day"))
                                with c1:
                                    max_day = dow_df.loc[dow_df['Effect'].idxmax()]
                                    st.write(f"""
                                    **Interpretation:**
                                    *   **{max_day['Day']}** is naturally your strongest day (+{max_day['Effect']:.1f}).
                                    *   The model *subtracts* this natural pattern before calculating Lift.
                                    """)
                            else:
                                st.write("Day of week effects not available.")

                        with tab3:
                            st.subheader("How did we create the Baseline?")
                            st.write("We used the following Control Geos to predict what WOULD have happened without your change:")
                            
                            if coeffs:
                                coeff_df = pd.DataFrame({
                                    "Control Geo": control_geos,
                                    "Weight": coeffs
                                }).sort_values(by="Weight", ascending=False)
                                st.bar_chart(coeff_df.set_index("Control Geo"))
                            
                        
                        with st.expander("Show Raw Results (JSON)"):
                            st.json(results)
                        
                    except Exception as e:
                        st.error(f"Experiment Failed: {e}")
                        
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("👋 Welcome to OpenLift! Upload your CSV file to simple drag-and-drop measurement.")

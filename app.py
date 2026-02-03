import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from pathlib import Path
from openlift.core.pipeline import run_geo_lift_df
from openlift.core.design import GeoMatcher, PowerAnalysis
from openlift.core.llm import LLMService

st.set_page_config(page_title="OpenLift UI", layout="wide")

st.title("OpenLift: Marketing Incrementality Platform")

import base64

# --- BRANDING: LOCAL POPPINS FONT ---
def load_local_font(font_path, font_name, font_weight=400):
    try:
        with open(font_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        return f"""
            @font-face {{
                font-family: '{font_name}';
                font-style: normal;
                font-weight: {font_weight};
                src: url(data:font/ttf;base64,{b64}) format('truetype');
            }}
        """
    except FileNotFoundError:
        return ""

# Load fonts from assets
font_css = ""
font_css += load_local_font("assets/Poppins-Regular.ttf", "Poppins", 400)
font_css += load_local_font("assets/Poppins-Bold.ttf", "Poppins", 600)
# Fallback if files missing (keep Google Fonts as backup or just rely on local)
# We will use local if available, otherwise fallback.

st.markdown(
    f"""
    <style>
    {font_css}
    
    html, body, [class*="css"]  {{
        font-family: 'Poppins', sans-serif;
    }}
    
    /* Customizing Headers */
    h1, h2, h3 {{
        font-weight: 600;
        color: #000000;
    }}
    
    /* Button Tweaks (Black Text on Yellow Button usually needed if streamlit defaults to white) */
    div.stButton > button:first-child {{
        font-weight: 600;
        color: #000000 !important; /* Force black text on yellow buttons */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# 1. Upload Data
st.sidebar.header("Global Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV (date, geo, outcome)", type=["csv"])

if uploaded_file is not None:
    @st.cache_data
    def load_csv(file):
        return pd.read_csv(file)
        
    df = load_csv(uploaded_file)
    
    st.sidebar.success(f"Loaded {len(df)} rows.")
    
    # LLM Config
    st.sidebar.divider()
    st.sidebar.subheader("🤖 AI Co-Pilot")
    llm_provider_selection = st.sidebar.selectbox("Provider", ["Gemini", "Ollama (Local)"], index=0)
    
    api_key = None
    model_name = None
    provider_key = "gemini"
    
    if llm_provider_selection == "Gemini":
        provider_key = "gemini"
        api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter Google AI Studio key.")
        model_name = "gemini-2.0-flash"
    else:
        provider_key = "ollama"
        model_name = st.sidebar.text_input("Local Model Name", value="llama3", help="Make sure you have this model installed via 'ollama pull llama3'")
        st.sidebar.info("Ensure Ollama is running (`ollama serve`).")
        # Cloud Warning
        # Cloud Warning
        is_cloud = False
        try:
            if hasattr(st, "context") and hasattr(st.context, "headers"):
                host = st.context.headers.get("host", "")
                if "streamlit.app" in str(host):
                    is_cloud = True
        except Exception:
            pass

        if is_cloud:
             st.sidebar.warning("⚠️ **Cloud Note:** Ollama (Local) will NOT work on Streamlit Cloud. Switch to Gemini.")

    llm = LLMService(provider=provider_key, api_key=api_key, model_name=model_name)
    
    # Global Config
    st.sidebar.header("Global Mapping")
    date_col = st.sidebar.selectbox("Date Column", df.columns, index=0)
    geo_col = st.sidebar.selectbox("Geo Column", df.columns, index=1)
    outcome_col = st.sidebar.selectbox("Outcome Metric (Y)", df.columns, index=2)
    
    # Cost/Input Metric
    cost_col = st.sidebar.selectbox("Input Metric (X) [Optional]", ["None"] + list(df.columns), index=0, help="The metric you changed/increased (e.g. Amount Spend, Impressions). Used to calculate ROI.")
    
    # Pre-processing dates
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        st.error(f"Error parsing date column: {e}")
        st.stop()
        
    min_date = df[date_col].min().date()
    max_date = df[date_col].max().date()
    all_geos = sorted(df[geo_col].unique().tolist())
    
    tab1, tab2, tab3 = st.tabs(["🚀 Experiment Runner", "🗺️ Geo Matcher", "⚡ Power Analysis"])
    
    # ==========================================
    # TAB 1: EXPERIMENT RUNNER (Existing Logic)
    # ==========================================
    with tab1:
        st.header("Run Impact Measurement")
        
        c1, c2 = st.columns(2)
        with c1:
            test_geo = st.selectbox("Test Geo", all_geos, key="run_test")
        with c2:
            control_geos = st.multiselect("Control Geos", [g for g in all_geos if g != test_geo], default=[g for g in all_geos if g != test_geo][:5], key="run_controls")
            
        col_dates = st.columns(4)
        pre_start = col_dates[0].date_input("Pre Start", min_date, min_value=min_date, max_value=max_date, key="run_pre_start")
        pre_end = col_dates[1].date_input("Pre End", min_date, min_value=min_date, max_value=max_date, key="run_pre_end")
        post_start = col_dates[2].date_input("Post Start", max_date, min_value=min_date, max_value=max_date, key="run_post_start")
        post_end = col_dates[3].date_input("Post End", max_date, min_value=min_date, max_value=max_date, key="run_post_end")
        
        if st.button("Run Measurement", type="primary"):
            if len(control_geos) < 2:
                st.error("Need at least 2 control geos.")
            else:
                with st.spinner("Running MCMC Sampling..."):
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
                        m = results['metrics']
                        
                        # --- ANALYZE INPUT (COST) CHANGE ---
                        input_change_pct = 0.0
                        input_change_abs = 0.0
                        has_input = cost_col != "None"
                        
                        if has_input:
                            # Filter for Test Geo
                            mask_geo = df[geo_col] == test_geo
                            
                            # Pre Period Stats
                            # Calculate daily stats by reindexing to ensure missing days are counted as 0
                            pre_range = pd.date_range(start=pre_start, end=pre_end, freq='D')
                            temp_geo_df = df[mask_geo].set_index(date_col).reindex(pre_range).fillna(0)
                            pre_val = temp_geo_df[cost_col].mean()
                            
                            # Post Period Stats
                            post_range = pd.date_range(start=post_start, end=post_end, freq='D')
                            temp_geo_post_df = df[mask_geo].set_index(date_col).reindex(post_range).fillna(0)
                            
                            post_sum = temp_geo_post_df[cost_col].sum()
                            post_mean = temp_geo_post_df[cost_col].mean()
                            
                            # Calculate Lift in Input (Simple Pre-Post difference, naive)
                            if pre_val > 0:
                                input_change_pct = ((post_mean - pre_val) / pre_val) * 100
                            # Total Incremental Spend approx = (Post Daily Avg - Pre Daily Avg) * Duration ?
                            # Or strict sum difference? 
                            # Lift typically implies "Rate" change.
                            # But Total Delta is often what matters for CPI.
                            # Let's use: Actual Post Spend - (Pre Daily Avg * Post Days)
                            input_change_abs = post_sum - (pre_val * len(post_range))
                        
                        st.divider()
                        # KPIs
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("Incremental Lift (Y)", f"{m['incremental_outcome_mean']:.1f}")
                        k2.metric("Lift % (Y)", f"{m['lift_pct_mean']:.1f}%")
                        k3.metric("Confidence", f"{m['p_positive']*100:.1f}%")
                        
                        if has_input:
                            k4.metric(
                                f"Input Shift ({cost_col})", 
                                f"{input_change_pct:+.1f}%",
                                f"Est. Delta: {input_change_abs:+.1f}"
                            )
                        else:
                            k4.metric("Input Shift", "-", "Select Input Col to see ROI")

                        # Efficiency / ROI
                        if has_input and m['incremental_outcome_mean'] > 0 and input_change_abs > 0:
                            cpi = input_change_abs / m['incremental_outcome_mean']
                            st.info(f"💰 **Efficiency:** You spent approx. **{input_change_abs:.1f}** more {cost_col} to get **{m['incremental_outcome_mean']:.1f}** more {outcome_col}.\n\n**Cost Per Incremental Result:** {cpi:.2f}")

                        # --- INSIGHTS ---
                        st.subheader("💡 Analysis & Insights")
                        
                        # Lift Insight
                        lift_sign = "POSITIVE" if m['incremental_outcome_mean'] > 0 else "NEGATIVE"
                        lift_quality = "SIGNIFICANT" if m['p_positive'] > 0.9 else "DIRECTIONAL"
                        
                        # Standard Rule-Based Insight (Fallback)
                        insight_text = f"""
                        **What happened?**
                        We observed a **{m['lift_pct_mean']:.1f}% {lift_sign.lower()} lift** in {outcome_col} during the test period. 
                        This means your campaign generated approximately **{m['incremental_outcome_mean']:.0f} extra {outcome_col}** that wouldn't have happened otherwise.
                        
                        **How sure are we?**
                        We are **{m['p_positive']*100:.1f}% confident** that this lift is real (not just random noise). 
                        """
                        
                        if m['p_positive'] > 0.95:
                            insight_text += "This is a **very strong result**. You can be confident in these numbers."
                        elif m['p_positive'] > 0.8:
                            insight_text += "This is a **promising result**, but there is still some chance (approx 20%) it could be noise."
                        else:
                            insight_text += "⚠️ **Caution:** This result is **not statistically significant**. The lift we see might just be random fluctuation."
                            
                        st.markdown(insight_text)
                        
                        # --- LLM SMART INSIGHT ---
                        if llm.is_available():
                            with st.spinner(f"Generating Analysis ({model_name})..."):
                                context = f"Test Geo: {test_geo}, Outcome: {outcome_col}, Pre-Period: {pre_start} to {pre_end}, Post-Period: {post_start} to {post_end}"
                                if has_input:
                                    context += f", Input Shift: {input_change_abs:.1f} ({cost_col})"
                                    
                                smart_insight = llm.get_experiment_insights(results, context_str=context)
                                st.info(f"🤖 **AI Co-Pilot Verdict:**\n\n{smart_insight}")
                        elif api_key: # Key entered but invalid?
                             st.warning("AI Key provided but client failed to initialize.")
                        else:
                             if provider_key == "gemini":
                                 st.markdown("*Tip: Enter Gemini API Key in sidebar for AI analysis.*")
                             else:
                                 st.warning("Ollama provider selected but not available.")
                        
                        st.json(results)
                        
                    except Exception as e:
                        st.error(f"Error: {e}")

    # ==========================================
    # TAB 2: GEO MATCHER
    # ==========================================
    with tab2:
        st.header("Find Best Control Markets")
        st.markdown("Use historical data to find markets that behave most similarly to your target.")
        
        c1, c2 = st.columns(2)
        with c1:
            target_geo = st.selectbox("Target Geo", all_geos, key="match_target")
        with c2:
            lookback = st.slider("Lookback Days", 30, 365, 90)
            
        if st.button("Find Matches"):
            matcher = GeoMatcher(df, date_col, geo_col, outcome_col)
            matches = matcher.find_controls(target_geo, lookback_days=lookback, n_controls=10)
            
            st.subheader(f"Top Matches for {target_geo}")
            
            # Insights for Matching
            best_match = matches[0][0]
            best_score = matches[0][1]
            st.info(f"""
            💡 **Recommendation:** 
            **{best_match}** is your best control market. It moves almost exactly like {target_geo}.
            
            **Why?** The "Distance Score" ({best_score:.2f}) measures similarity. Lower is better. 
            Markets with low scores are "twins" — they trend up and down together, making them perfect for comparing against.
            """)
            
            match_df = pd.DataFrame(matches, columns=["Geo", "Distance Score"])
            st.dataframe(match_df)
            
            # Visual check
            st.subheader("Visual Comparison")
            # Plot target vs top 3 matches
            top_3 = [m[0] for m in matches[:3]]
            chart_geos = [target_geo] + top_3
            
            chart_data = df[df[geo_col].isin(chart_geos)].pivot(index=date_col, columns=geo_col, values=outcome_col)
            # Filter for lookback
            chart_data = chart_data.iloc[-lookback:]
            
            st.line_chart(chart_data)

    # ==========================================
    # TAB 3: POWER ANALYSIS
    # ==========================================
    with tab3:
        st.header("Pre-Flight Power Simulator")
        st.markdown("Estimate your chance of detecting a lift BEFORE you run the test.")
        
        c1, c2 = st.columns(2)
        with c1:
            pa_target = st.selectbox("Test Geo", all_geos, key="pa_target")
        with c2:
            pa_controls = st.multiselect("Control Geos", [g for g in all_geos if g != pa_target], key="pa_controls")
            
        c3, c4, c5 = st.columns(3)
        lift_est = c3.number_input("Expected Lift %", 0.01, 1.0, 0.10, 0.01)
        duration = c4.number_input("Test Duration (Days)", 7, 90, 30)
        sims = c5.number_input("Simulations", 10, 100, 20)
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            lookback = st.slider("Lookback Days (Training Data)", 14, 180, 60, help="Days before test start to train the model.")
        
        if st.button("Run Simulation"):
            if not pa_controls:
                st.error("Select controls.")
            else:
                with st.spinner(f"Simulating {sims} experiments..."):
                    pa = PowerAnalysis(df, date_col, geo_col, outcome_col)
                    res = pa.simulate_power(
                        pa_target,
                        pa_controls,
                        effect_size_pct=lift_est,
                        test_duration_days=duration,
                        simulations=sims,
                        lookback_days=lookback
                    )
                    
                power = res['power']
                
                st.divider()
                st.subheader("💡 Pre-Flight Check")
                
                st.metric("Estimated Probability of Success", f"{power*100:.1f}%")
                
                if power > 0.8:
                    st.success(f"""
                    **✅ You are good to go!**
                    
                    If your campaign actually drives a **{lift_est*100:.0f}% lift**, our model has an **{power*100:.0f}% chance** of detecting it successfully.
                    
                    **What this means:** Your historical data is stable enough, and your test duration ({duration} days) is long enough to find this signal.
                    """)
                else:
                    # Low Power Case: Recommend MDE
                    st.warning(f"⚠️ **Low Power ({power*100:.0f}%)**: A {lift_est*100:.0f}% lift is hard to detect.")
                    
                    with st.spinner("Calculating Minimum Detectable Spend..."):
                        mde_lift = pa.find_mde(pa_target, pa_controls, duration, lookback_days=lookback)
                    
                    if mde_lift > 0:
                        rec_text = f"""
                        **💡 Recommendation:** 
                        To reach **80% Confidence**, you need a target lift of at least **{mde_lift*100:.0f}%**.
                        """
                        
                        # Calculate Spend for MDE
                        if cost_col != "None":
                             est_mde = pa.estimate_required_input(
                                pa_target, 
                                cost_col, 
                                mde_lift, 
                                duration, 
                                lookback
                            )
                             if est_mde and est_mde['required_input'] > 0:
                                 rec_text += f"\n\n💰 **Required Investment:** approx. **{est_mde['required_input']:.0f} {cost_col}**."
                        
                        st.info(rec_text)
                    else:
                        st.error("Even a 50% lift would be hard to detect in this geo. Consider a longer duration.")

                # --- INPUT ESTIMATION (User's Current Input) ---
                if cost_col != "None":
                    est = pa.estimate_required_input(
                        pa_target, 
                        cost_col, 
                        lift_est, 
                        duration, 
                        lookback
                    )
                    
                    if est and est['required_input'] > 0:
                        st.info(f"""
                        💰 **Investment Estimate**
                        
                        To achieve this **{lift_est*100:.0f}% lift** (+{est['target_lift_abs']:.0f} {outcome_col}), 
                        you likely need to adding approx. **{est['required_input']:.0f} {cost_col}** of investment.
                        
                        *(Based on historical efficiency of {est['cpr']:.2f} {cost_col} per {outcome_col} in {pa_target})*
                        """)
                    
                # --- LLM SMART INSIGHT ---
                if llm.is_available():
                        with st.spinner("Generating AI Recommendations..."):
                            smart_power = llm.get_power_analysis_insights(res)
                            st.info(f"🤖 **AI Co-Pilot Recommendations:**\n\n{smart_power}")

else:
    st.info("Please upload a CSV file to begin.")

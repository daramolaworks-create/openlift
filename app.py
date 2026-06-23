import streamlit as st
import pandas as pd
import numpy as np
from openlift.core.pipeline import run_geo_lift_df
from openlift.core.design import GeoMatcher, PowerAnalysis
from openlift.core.llm import LLMService
from openlift.core.multi_cell import (
    MultiCellExperimentConfig,
    CellConfig,
    run_multi_cell_experiment,
)
from openlift.connectors import GoogleSheetsConnector, GoogleAdsConnector, MetaAdsConnector
from openlift.core.diagnostics import DataValidator
from openlift.core.economics import calculate_economics, recommend_budget, simulate_budget_scenarios, build_payback_curve
from openlift.core.evidence import calculate_evidence_strength
from openlift.core.report_generator import MarkdownReportGenerator
from openlift.core.decision import build_decision_summary
from openlift.core.creative import analyze_creative_lift, infer_creative_columns
from openlift.core.experiment_designer import recommend_next_experiment
from openlift.core.memory import ExperimentRegistry
# Check which optional connectors are available
HAS_GOOGLE_ADS = GoogleAdsConnector is not None
HAS_META_ADS = MetaAdsConnector is not None

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
font_css += load_local_font("assets/Poppins-ExtraBold.ttf", "Poppins", 700)
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

# 1. Data Source Selection
st.sidebar.header("Data Source")

# Build available sources
_sources = ["CSV Upload", "Google Sheets"]
if HAS_GOOGLE_ADS:
    _sources.append("Google Ads")
if HAS_META_ADS:
    _sources.append("Meta Ads")

data_source = st.sidebar.radio("Load data from:", _sources, index=0, horizontal=True)

uploaded_file = None
df = None

if data_source == "CSV Upload":
    uploaded_file = st.sidebar.file_uploader("Upload CSV (date, geo, outcome)", type=["csv"])

# ------------------------------------------------------------------
# Google Sheets Connector
# ------------------------------------------------------------------
elif data_source == "Google Sheets":
    st.sidebar.markdown("---")
    gs_method = st.sidebar.selectbox("Auth Method", ["API Key (Public Sheets)", "Service Account JSON"], key="gs_auth")
    gs_sheet_url = st.sidebar.text_input("Google Sheet URL", help="Full URL of the Google Spreadsheet")
    gs_worksheet = st.sidebar.text_input("Worksheet Name (optional)", help="Leave blank for first sheet")
    gs_date_col = st.sidebar.text_input("Date column name in sheet", value="date")
    gs_geo_col = st.sidebar.text_input("Geo column name in sheet", value="geo")
    gs_outcome_col = st.sidebar.text_input("Outcome column name in sheet", value="outcome")

    if gs_method == "API Key (Public Sheets)":
        gs_api_key = st.sidebar.text_input("Google API Key", type="password")
    else:
        gs_sa_file = st.sidebar.file_uploader("Service Account JSON", type=["json"], key="gs_sa")

    if st.sidebar.button("Connect & Fetch", key="gs_connect"):
        connector = GoogleSheetsConnector()
        creds = {
            "sheet_url": gs_sheet_url,
            "date_col": gs_date_col,
            "geo_col": gs_geo_col,
            "outcome_col": gs_outcome_col,
        }
        if gs_worksheet:
            creds["worksheet"] = gs_worksheet

        if gs_method == "API Key (Public Sheets)" and gs_api_key:
            creds["api_key"] = gs_api_key
        elif gs_method == "Service Account JSON" and gs_sa_file:
            import json as _json
            creds["service_account_info"] = _json.loads(gs_sa_file.read())
        else:
            st.sidebar.error("Provide credentials.")
            st.stop()

        with st.sidebar:
            with st.spinner("Connecting to Google Sheets..."):
                if connector.authenticate(creds):
                    fetched = connector.fetch_data(
                        start_date="2000-01-01",
                        end_date="2099-12-31",
                        geo_col=gs_geo_col,
                        outcome_col=gs_outcome_col,
                    )
                    st.session_state["connector_df"] = fetched
                    st.sidebar.success(f"Fetched {len(fetched)} rows from Google Sheets.")
                else:
                    st.sidebar.error("Connection failed. Check URL and credentials.")

# ------------------------------------------------------------------
# Google Ads Connector
# ------------------------------------------------------------------
elif data_source == "Google Ads":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Google Ads Credentials")
    ga_dev_token = st.sidebar.text_input("Developer Token", type="password", key="ga_dev")
    ga_client_id = st.sidebar.text_input("Client ID", key="ga_cid")
    ga_client_secret = st.sidebar.text_input("Client Secret", type="password", key="ga_cs")
    ga_refresh_token = st.sidebar.text_input("Refresh Token", type="password", key="ga_rt")
    ga_customer_id = st.sidebar.text_input("Customer ID (10 digits, no dashes)", key="ga_custid")
    ga_login_customer_id = st.sidebar.text_input("Login Customer ID (MCC, optional)", key="ga_login")
    ga_geo_level = st.sidebar.selectbox("Geo Level", ["region", "city", "country"], key="ga_geo_level")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Date Range")
    ga_start = st.sidebar.text_input("Start Date (YYYY-MM-DD)", value="2024-01-01", key="ga_start")
    ga_end = st.sidebar.text_input("End Date (YYYY-MM-DD)", value="2024-12-31", key="ga_end")
    ga_outcome = st.sidebar.selectbox("Primary Outcome", ["conversions", "spend", "clicks", "impressions"], key="ga_outcome")

    if st.sidebar.button("Connect & Fetch", key="ga_connect"):
        connector = GoogleAdsConnector()
        creds = {
            "developer_token": ga_dev_token,
            "client_id": ga_client_id,
            "client_secret": ga_client_secret,
            "refresh_token": ga_refresh_token,
            "customer_id": ga_customer_id,
            "geo_level": ga_geo_level,
        }
        if ga_login_customer_id:
            creds["login_customer_id"] = ga_login_customer_id

        with st.sidebar:
            with st.spinner("Connecting to Google Ads..."):
                if connector.authenticate(creds):
                    try:
                        fetched = connector.fetch_data(
                            start_date=ga_start,
                            end_date=ga_end,
                            outcome_col=ga_outcome,
                        )
                        st.session_state["connector_df"] = fetched
                        st.sidebar.success(f"Fetched {len(fetched)} rows from Google Ads.")
                    except Exception as e:
                        st.sidebar.error(f"Fetch failed: {e}")
                else:
                    st.sidebar.error("Auth failed. Check credentials.")

# ------------------------------------------------------------------
# Meta Ads Connector
# ------------------------------------------------------------------
elif data_source == "Meta Ads":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Meta Ads Credentials")
    meta_token = st.sidebar.text_input("Access Token", type="password", key="meta_token", help="Long-lived user or system user token")
    meta_account = st.sidebar.text_input("Ad Account ID", key="meta_account", help="e.g. act_123456789")
    meta_app_id = st.sidebar.text_input("App ID (optional)", value="0", key="meta_appid")
    meta_app_secret = st.sidebar.text_input("App Secret (optional)", type="password", key="meta_secret")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Date Range")
    meta_start = st.sidebar.text_input("Start Date (YYYY-MM-DD)", value="2024-01-01", key="meta_start")
    meta_end = st.sidebar.text_input("End Date (YYYY-MM-DD)", value="2024-12-31", key="meta_end")
    meta_outcome = st.sidebar.selectbox("Primary Outcome", ["conversions", "spend", "clicks", "impressions"], key="meta_outcome")

    if st.sidebar.button("Connect & Fetch", key="meta_connect"):
        connector = MetaAdsConnector()
        creds = {
            "access_token": meta_token,
            "ad_account_id": meta_account,
            "app_id": meta_app_id,
            "app_secret": meta_app_secret,
        }

        with st.sidebar:
            with st.spinner("Connecting to Meta Ads..."):
                if connector.authenticate(creds):
                    try:
                        fetched = connector.fetch_data(
                            start_date=meta_start,
                            end_date=meta_end,
                            outcome_col=meta_outcome,
                        )
                        st.session_state["connector_df"] = fetched
                        st.sidebar.success(f"Fetched {len(fetched)} rows from Meta Ads.")
                    except Exception as e:
                        st.sidebar.error(f"Fetch failed: {e}")
                else:
                    st.sidebar.error("Auth failed. Check token and account ID.")

# Load from connector session state
if "connector_df" in st.session_state and df is None:
    df = st.session_state["connector_df"]

if uploaded_file is not None:
    @st.cache_data
    def load_csv(file):
        return pd.read_csv(file)
    df = load_csv(uploaded_file)

if df is not None:
    
    st.sidebar.success(f"Loaded {len(df)} rows.")
    
    # LLM Config
    st.sidebar.divider()
    st.sidebar.subheader("🤖 AI Co-Pilot")
    llm_provider_selection = st.sidebar.selectbox(
        "Provider",
        ["Gemini", "Claude (Anthropic)", "DeepSeek (Reasoner)", "Ollama (Local)"],
        index=0,
    )

    api_key = None
    model_name = None
    provider_key = "gemini"

    if llm_provider_selection == "Gemini":
        provider_key = "gemini"
        api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter Google AI Studio key.")
        model_name = st.sidebar.text_input("Gemini Model", value="gemini-2.5-flash", help="e.g. gemini-2.5-flash, gemini-1.5-pro")
    elif llm_provider_selection == "Claude (Anthropic)":
        provider_key = "claude"
        api_key = st.sidebar.text_input("Anthropic API Key", type="password", help="Enter your Anthropic API key.")
        model_name = st.sidebar.text_input("Claude Model", value="claude-sonnet-4-6", help="e.g. claude-sonnet-4-6, claude-opus-4-8")
    elif llm_provider_selection == "DeepSeek (Reasoner)":
        provider_key = "deepseek"
        api_key = st.sidebar.text_input("DeepSeek API Key", type="password", help="Enter DeepSeek API key.")
        model_name = st.sidebar.text_input("DeepSeek Model", value="deepseek-reasoner", help="defaults to deepseek-reasoner")
    else:
        provider_key = "ollama"
        model_name = st.sidebar.text_input("Local Model Name", value="llama3", help="Make sure you have this model installed via 'ollama pull llama3'")
        st.sidebar.info("Ensure Ollama is running (`ollama serve`).")
        is_cloud = False
        try:
            if hasattr(st, "context") and hasattr(st.context, "headers"):
                host = st.context.headers.get("host", "")
                if "streamlit.app" in str(host):
                    is_cloud = True
        except Exception:
            pass
        if is_cloud:
            st.sidebar.warning("⚠️ **Cloud Note:** Ollama (Local) will NOT work on Streamlit Cloud. Switch to Gemini or Claude.")

    llm = LLMService(provider=provider_key, api_key=api_key, model_name=model_name)
    
    # Global Config
    st.sidebar.header("Global Mapping")
    date_col = st.sidebar.selectbox("Date Column", df.columns, index=0)
    geo_col = st.sidebar.selectbox("Geo Column", df.columns, index=1)
    outcome_col = st.sidebar.selectbox("Outcome Metric (Y)", df.columns, index=2)
    
    # Cost/Input Metric
    cost_col = st.sidebar.selectbox("Input Metric (X) [Optional]", ["None"] + list(df.columns), index=0, help="The metric you changed/increased (e.g. Amount Spend, Impressions). Used to calculate ROI.")
    
    st.sidebar.divider()
    st.sidebar.subheader("Economics")
    margin_pct = st.sidebar.number_input("Gross Margin (%)", min_value=0.0, max_value=100.0, value=100.0) / 100.0
    ltv = st.sidebar.number_input("Customer LTV", min_value=0.0, value=1.0, help="If Outcome is conversions, what is the value of one conversion?")
    
    # Pre-processing dates
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        st.error(f"Error parsing date column: {e}")
        st.stop()
        
    min_date = df[date_col].min().date()
    max_date = df[date_col].max().date()
    all_geos = sorted(df[geo_col].unique().tolist())
    
    # --- DATA QUALITY DIAGNOSTICS ---
    validator = DataValidator(df, date_col, geo_col, outcome_col, cost_col)
    dq_result = validator.evaluate()
    with st.expander(f"📊 Data Quality Score: {dq_result['score']}/100 ({dq_result['status']})"):
        if dq_result['warnings']:
            for w in dq_result['warnings']:
                st.warning(w)
        else:
            st.success("Data looks clean and ready for modeling.")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🚀 Experiment Runner",
        "🗺️ Geo Matcher",
        "⚡ Power Analysis",
        "🧪 Multi-Cell",
        "🎨 Creative Lift",
        "🧭 Next Experiment",
        "📚 Scorecard",
    ])
    
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
        
        # --- External Covariates ---
        from openlift.core.covariates import SUPPORTED_COUNTRIES, GEO_COORDINATES
        
        with st.expander("🌦️ External Covariates (Holidays & Weather)", expanded=False):
            st.markdown(
                "Adding external covariates helps the model control for holidays and weather shocks, "
                "resulting in **tighter confidence intervals** and a **lower MDE**."
            )
            cov_c1, cov_c2 = st.columns(2)
            
            with cov_c1:
                use_holidays = st.toggle("Control for Holidays", value=False, key="use_holidays")
                if use_holidays:
                    country_options = [f"{code} — {name}" for code, name in SUPPORTED_COUNTRIES.items()]
                    selected_country = st.selectbox("Country", country_options, index=0, key="holiday_country")
                    cov_country_code = selected_country.split(" — ")[0]
                else:
                    cov_country_code = None
            
            with cov_c2:
                use_weather = st.toggle("Control for Weather", value=False, key="use_weather")
                if use_weather:
                    # Try auto-detect from test geo
                    auto_coords = GEO_COORDINATES.get(test_geo.lower().strip()) if test_geo else None
                    default_lat = auto_coords[0] if auto_coords else 6.45
                    default_lon = auto_coords[1] if auto_coords else 3.40
                    
                    if auto_coords:
                        st.success(f"Auto-detected: {test_geo} → ({default_lat}, {default_lon})")
                    
                    cov_lat = st.number_input("Latitude", value=default_lat, format="%.4f", key="cov_lat")
                    cov_lon = st.number_input("Longitude", value=default_lon, format="%.4f", key="cov_lon")
                else:
                    cov_lat = None
                    cov_lon = None
        
        if st.button("Run Measurement", type="primary"):
            if len(control_geos) < 2:
                st.error("Need at least 2 control geos.")
            else:
                spinner_msg = "Running MCMC Sampling"
                if use_holidays or use_weather:
                    extras = []
                    if use_holidays:
                        extras.append("holidays")
                    if use_weather:
                        extras.append("weather")
                    spinner_msg += f" (with {', '.join(extras)} covariates)"
                spinner_msg += "..."
                
                with st.spinner(spinner_msg):
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
                            outcome_col=outcome_col,
                            country_code=cov_country_code,
                            latitude=cov_lat,
                            longitude=cov_lon,
                        )
                        m = results['metrics']
                        dq_run = DataValidator(
                            df,
                            date_col,
                            geo_col,
                            outcome_col,
                            cost_col,
                            pre_start=str(pre_start),
                            pre_end=str(pre_end),
                            post_start=str(post_start),
                            post_end=str(post_end),
                        ).evaluate()

                        match_score = None
                        try:
                            matcher = GeoMatcher(df, date_col, geo_col, outcome_col)
                            control_matches = matcher.find_controls(
                                test_geo,
                                pool_geos=control_geos,
                                lookback_days=max(14, (pre_end - pre_start).days + 1),
                                n_controls=len(control_geos),
                            )
                            if control_matches:
                                match_score = float(np.mean([score for _, score in control_matches]))
                                results["match_quality"] = {
                                    "method": "dtw",
                                    "average_distance": match_score,
                                    "control_scores": [
                                        {"geo": geo, "distance": float(score)}
                                        for geo, score in control_matches
                                    ],
                                }
                        except Exception as match_error:
                            results["match_quality"] = {"error": str(match_error)}
                        
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
                        # --- ECONOMICS & EVIDENCE ---
                        outcome_is_revenue = True if "revenue" in outcome_col.lower() else False
                        eco = calculate_economics(
                            incremental_outcome=m['incremental_outcome_mean'],
                            input_change_abs=input_change_abs,
                            outcome_is_revenue=outcome_is_revenue,
                            margin_pct=margin_pct,
                            ltv=ltv
                        )
                        
                        relative_hdi_width = (m.get('incremental_outcome_hdi_90', [0,0])[1] - m.get('incremental_outcome_hdi_90', [0,0])[0]) / m['incremental_outcome_mean'] if m['incremental_outcome_mean'] > 0 else 2.0
                        evidence = calculate_evidence_strength(m['p_positive'], match_score=match_score, relative_hdi_width=relative_hdi_width)
                        recommendation = recommend_budget(m['p_positive'], eco.get("incremental_roas"), eco.get("incremental_profit"))
                        decision = build_decision_summary(
                            m,
                            economics=eco,
                            data_quality=dq_run,
                            match_score=match_score,
                            pre_period_days=(pre_end - pre_start).days + 1,
                            post_period_days=(post_end - post_start).days + 1,
                        )
                        results["decision"] = decision
                        ExperimentRegistry().add_result(results)

                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("Incremental Lift (Y)", f"{m['incremental_outcome_mean']:.1f}")
                        k2.metric("Lift % (Y)", f"{m['lift_pct_mean']:.1%}")
                        k3.metric("Evidence Strength", decision["evidence_strength"])
                        
                        if has_input:
                            roas_str = f"{eco.get('incremental_roas', 0):.2f}" if eco.get('incremental_roas') else "N/A"
                            prof_str = f"{eco.get('incremental_profit', 0):.0f}" if eco.get('incremental_profit') else "N/A"
                            k4.metric(f"Inc. ROAS / Profit", f"{roas_str}", f"{prof_str} profit")
                        else:
                            k4.metric("Input Shift", "-", "Select Input Col to see ROI")

                        if m.get("posterior_lift_distribution"):
                            st.subheader("Posterior Lift Distribution")
                            posterior_preview = pd.DataFrame({
                                "Incremental lift": m["posterior_lift_distribution"]["samples_preview"]
                            })
                            st.bar_chart(posterior_preview)

                        # Efficiency / ROI
                        if has_input and m['incremental_outcome_mean'] > 0 and input_change_abs > 0:
                            st.info(f"💰 **Efficiency:** You spent approx. **{input_change_abs:.1f}** more {cost_col} to get **{m['incremental_outcome_mean']:.1f}** more {outcome_col}.")
                            scenarios = simulate_budget_scenarios(
                                current_input=post_sum,
                                incremental_roas=eco.get("incremental_roas"),
                                incremental_cac=eco.get("incremental_cac"),
                                incremental_profit=eco.get("incremental_profit"),
                                p_positive=m["p_positive"],
                                risk_tolerance="medium",
                            )
                            scenario_df = pd.DataFrame(scenarios)
                            scenario_df["change_pct"] = scenario_df["change_pct"].map(lambda x: f"{x:+.0%}")
                            st.dataframe(
                                scenario_df[
                                    [
                                        "change_pct",
                                        "new_input",
                                        "input_delta",
                                        "confidence_weight",
                                        "expected_incremental_revenue",
                                        "expected_incremental_profit",
                                    ]
                                ],
                                use_container_width=True,
                            )
                            incremental_revenue = (
                                m["incremental_outcome_mean"]
                                if outcome_is_revenue
                                else m["incremental_outcome_mean"] * ltv
                            )
                            payback_curve = pd.DataFrame(
                                build_payback_curve(
                                    incremental_revenue=incremental_revenue,
                                    input_change_abs=input_change_abs,
                                    margin_pct=margin_pct,
                                )
                            )
                            if not payback_curve.empty:
                                st.subheader("LTV / Payback Curve")
                                st.line_chart(payback_curve.set_index("month")[["cumulative_gross_profit", "remaining_payback"]])
                                if eco.get("payback_period_months") is not None:
                                    st.caption(f"Estimated payback period: {eco['payback_period_months']:.1f} months")

                        # --- COVARIATE EFFECTS ---
                        if m.get("covariate_effects"):
                            st.subheader("🌦️ Covariate Effects")
                            cov_effects = m["covariate_effects"]
                            cov_cols = st.columns(len(cov_effects))
                            for idx, (cov_name, effect) in enumerate(cov_effects.items()):
                                icon = "📅" if "holiday" in cov_name else "🌡️" if "temp" in cov_name else "🌧️" if "precip" in cov_name else "📊"
                                cov_cols[idx].metric(
                                    f"{icon} {cov_name}",
                                    f"{effect:+.3f}",
                                    help=f"Learned effect of {cov_name} on daily {outcome_col}."
                                )

                        # --- INSIGHTS ---
                        st.subheader("💡 Analysis & Insights")
                        st.markdown(f"**Decision:** {decision['decision']} ({decision['scale_range']})")
                        st.markdown(f"**Recommendation:** {decision['recommendation']}")
                        with st.expander("Limitations & Next Action"):
                            for item in decision["limitations"]:
                                st.warning(item)
                            st.info(decision["next_action"])
                        
                        # Standard Rule-Based Insight (Fallback)
                        lift_sign = "POSITIVE" if m['incremental_outcome_mean'] > 0 else "NEGATIVE"
                        insight_text = f"""
                        **What happened?**
                        We observed a **{m['lift_pct_mean']:.1%} {lift_sign.lower()} lift** in {outcome_col} during the test period. 
                        This means your campaign generated approximately **{m['incremental_outcome_mean']:.0f} extra {outcome_col}** that wouldn't have happened otherwise.
                        
                        **How sure are we?**
                        We are **{m['p_positive']*100:.1f}% confident** that this lift is real.
                        """
                        st.markdown(insight_text)
                        
                        # --- REPORT EXPORT ---
                        report_gen = MarkdownReportGenerator(
                            experiment_name=f"Geo-Lift on {test_geo}",
                            test_geo=test_geo,
                            control_geos=control_geos,
                            metrics=m,
                            evidence=evidence,
                            economics=eco,
                            recommendation=recommendation,
                            data_quality=dq_run,
                            decision=decision,
                        )
                        md_report = report_gen.generate()
                        html_report = report_gen.generate_html()
                        st.download_button("📥 Download Executive Report", md_report, file_name=f"openlift_report_{test_geo}.md", mime="text/markdown")
                        st.download_button("📥 Download HTML Report", html_report, file_name=f"openlift_report_{test_geo}.html", mime="text/html")
                        
                        # --- LLM SMART INSIGHT ---
                        if llm.is_available():
                            with st.spinner(f"Generating Analysis ({model_name})..."):
                                context = f"Test Geo: {test_geo}, Outcome: {outcome_col}, Pre-Period: {pre_start} to {pre_end}, Post-Period: {post_start} to {post_end}"
                                if has_input:
                                    context += f", Input Shift: {input_change_abs:.1f} ({cost_col})"
                                    
                                smart_insight = llm.get_experiment_insights(results, context_str=context)
                                if isinstance(smart_insight, dict) and smart_insight.get("reasoning"):
                                    with st.expander("🧠 AI Thought Process (DeepSeek Reasoner)"):
                                        st.write(smart_insight["reasoning"])
                                    st.info(f"🤖 **AI Co-Pilot Verdict:**\n\n{smart_insight['content']}")
                                else:
                                    content = smart_insight['content'] if isinstance(smart_insight, dict) else smart_insight
                                    st.info(f"🤖 **AI Co-Pilot Verdict:**\n\n{content}")
                        elif api_key: # Key entered but invalid?
                             st.warning("AI Key provided but client failed to initialize.")
                        else:
                             if provider_key == "gemini":
                                 st.markdown("*Tip: Enter Gemini API Key in sidebar for AI analysis.*")
                             elif provider_key == "deepseek":
                                 st.markdown("*Tip: Enter DeepSeek API Key in sidebar for AI analysis.*")
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

            holdouts = matcher.recommend_holdouts([target_geo], lookback_days=lookback, n_holdouts=5)
            if holdouts:
                st.subheader("Recommended Holdout Group")
                st.dataframe(pd.DataFrame(holdouts), use_container_width=True)
            
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
                rec_duration = pa.recommend_duration(
                    pa_target,
                    pa_controls,
                    lift_est,
                    target_power=0.8,
                    max_duration=90,
                )
                
                st.divider()
                st.subheader("💡 Pre-Flight Check")
                
                st.metric("Estimated Probability of Success", f"{power*100:.1f}%")
                if rec_duration > 0:
                    st.metric("Recommended Duration", f"{rec_duration} days")
                
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
                            if isinstance(smart_power, dict) and smart_power.get("reasoning"):
                                with st.expander("🧠 AI Thought Process (DeepSeek Reasoner)"):
                                    st.write(smart_power["reasoning"])
                                st.info(f"🤖 **AI Co-Pilot Recommendations:**\n\n{smart_power['content']}")
                            else:
                                content = smart_power['content'] if isinstance(smart_power, dict) else smart_power
                                st.info(f"🤖 **AI Co-Pilot Recommendations:**\n\n{content}")

    # ==========================================
    # TAB 4: MULTI-CELL EXPERIMENT
    # ==========================================
    with tab4:
        st.header("Multi-Cell / Cross-Channel Experiment")
        st.markdown(
            "Test **multiple channels or campaigns simultaneously** against a shared holdout group. "
            "Compare which channel drives more incremental lift and detect synergy effects."
        )

        # --- Cell definition ---
        st.subheader("Define Cells")
        num_cells = st.number_input("Number of treatment cells", 2, 8, 2, key="mc_num_cells")

        cell_configs = []
        cell_cols = st.columns(min(int(num_cells), 4))
        used_geos = set()

        for i in range(int(num_cells)):
            col = cell_cols[i % len(cell_cols)]
            with col:
                label = st.text_input(f"Cell {i+1} Label", value=chr(65 + i), key=f"mc_label_{i}")
                name = st.text_input(f"Cell {i+1} Name", value=f"Channel {i+1}", key=f"mc_name_{i}")
                available = [g for g in all_geos if g not in used_geos]
                geos = st.multiselect(f"Cell {i+1} Test Geos", available, key=f"mc_geos_{i}")
                used_geos.update(geos)
                cell_configs.append({"name": name, "label": label, "test_geos": geos})

        # --- Shared controls ---
        st.subheader("Shared Holdout (Control Group)")
        remaining_geos = [g for g in all_geos if g not in used_geos]
        mc_controls = st.multiselect(
            "Control Geos",
            remaining_geos,
            default=remaining_geos[:min(5, len(remaining_geos))],
            key="mc_controls",
        )

        # --- Date ranges ---
        st.subheader("Experiment Periods")
        mc_date_cols = st.columns(4)
        mc_pre_start = mc_date_cols[0].date_input("Pre Start", min_date, min_value=min_date, max_value=max_date, key="mc_pre_start")
        mc_pre_end = mc_date_cols[1].date_input("Pre End", min_date, min_value=min_date, max_value=max_date, key="mc_pre_end")
        mc_post_start = mc_date_cols[2].date_input("Post Start", max_date, min_value=min_date, max_value=max_date, key="mc_post_start")
        mc_post_end = mc_date_cols[3].date_input("Post End", max_date, min_value=min_date, max_value=max_date, key="mc_post_end")

        # --- Run ---
        if st.button("Run Multi-Cell Experiment", type="primary", key="mc_run"):
            # Validate
            valid = True
            for i, cc in enumerate(cell_configs):
                if not cc["test_geos"]:
                    st.error(f"Cell {cc['label']} has no test geos.")
                    valid = False
            if len(mc_controls) < 2:
                st.error("Need at least 2 control geos.")
                valid = False

            if valid:
                with st.spinner("Running Multi-Cell MCMC (this may take a moment)..."):
                    try:
                        mc_config = MultiCellExperimentConfig(
                            name="multi_cell_experiment",
                            cells=[CellConfig(**cc) for cc in cell_configs],
                            control_geos=mc_controls,
                            pre_period={"start_date": mc_pre_start, "end_date": mc_pre_end},
                            post_period={"start_date": mc_post_start, "end_date": mc_post_end},
                        )

                        mc_results = run_multi_cell_experiment(
                            df=df,
                            config=mc_config,
                            date_col=date_col,
                            geo_col=geo_col,
                            outcome_col=outcome_col,
                        )

                        # --- Per-Cell Results ---
                        st.divider()
                        st.subheader("📊 Per-Cell Results")
                        result_cols = st.columns(len(cell_configs))
                        for idx, (label, result) in enumerate(mc_results["cells"].items()):
                            with result_cols[idx % len(result_cols)]:
                                if "error" in result:
                                    st.error(f"**{label}** — {result['error']}")
                                else:
                                    m = result["metrics"]
                                    st.markdown(f"### {label}: {result.get('cell_name', '')}")
                                    st.metric("Incremental Lift", f"{m['incremental_outcome_mean']:.1f}")
                                    st.metric("Lift %", f"{m['lift_pct_mean']:.1%}")
                                    st.metric("Confidence", f"{m['p_positive']*100:.0f}%")
                                    st.caption(f"Geos: {', '.join(result.get('cell_test_geos', []))}")

                        # --- Pairwise Comparisons ---
                        st.divider()
                        st.subheader("⚖️ Pairwise Comparisons")
                        for pair_key, comp in mc_results["comparisons"].items():
                            if "error" in comp:
                                st.warning(f"{pair_key}: {comp['error']}")
                                continue
                            winner = comp.get("winner", "Inconclusive")
                            conf = comp.get("confidence_level", "low")
                            delta = comp.get("absolute_delta", 0)

                            emoji = "🏆" if conf == "high" else "📊" if conf == "moderate" else "❓"
                            st.info(
                                f"{emoji} **{pair_key.replace('_vs_', ' vs ')}** — "
                                f"Winner: **{winner}** ({conf} confidence) | "
                                f"Absolute delta: {delta:.1f}"
                            )

                        # --- Synergy ---
                        if mc_results.get("synergy"):
                            st.divider()
                            st.subheader("🔗 Synergy Analysis")
                            syn = mc_results["synergy"]
                            if syn["is_super_additive"]:
                                st.success(
                                    f"✅ **Super-additive synergy detected!** "
                                    f"Combined cell '{syn['combined_cell']}' achieved "
                                    f"**{syn['synergy_pct']:.1f}% more lift** than the sum of individual cells."
                                )
                            else:
                                st.warning(
                                    f"⚠️ No positive synergy. Combined lift ({syn['combined_lift']:.1f}) "
                                    f"≤ sum of individual lifts ({syn['sum_individual_lifts']:.1f})."
                                )

                        # --- Comparison Bar Chart ---
                        st.divider()
                        st.subheader("📈 Lift Comparison")
                        chart_data = []
                        for label, result in mc_results["cells"].items():
                            if "error" not in result:
                                m = result["metrics"]
                                chart_data.append({
                                    "Cell": f"{label}: {result.get('cell_name', '')}",
                                    "Incremental Lift": m["incremental_outcome_mean"],
                                    "Lift %": m["lift_pct_mean"] * 100,
                                })
                        if chart_data:
                            chart_df = pd.DataFrame(chart_data).set_index("Cell")
                            st.bar_chart(chart_df["Incremental Lift"])

                        # --- AI Insight ---
                        if llm.is_available():
                            with st.spinner("Generating cross-channel strategy..."):
                                mc_insight = llm.get_multi_cell_insights(mc_results)
                                if isinstance(mc_insight, dict) and mc_insight.get("reasoning"):
                                    with st.expander("🧠 AI Thought Process (DeepSeek Reasoner)"):
                                        st.write(mc_insight["reasoning"])
                                    st.info(f"🤖 **AI Channel Strategy:**\n\n{mc_insight['content']}")
                                else:
                                    content = mc_insight["content"] if isinstance(mc_insight, dict) else mc_insight
                                    st.info(f"🤖 **AI Channel Strategy:**\n\n{content}")

                        # Raw JSON
                        with st.expander("Raw Results JSON"):
                            st.json(mc_results)

                    except Exception as e:
                        st.error(f"Multi-Cell Error: {e}")

    # ==========================================
    # TAB 5: CREATIVE LIFT
    # ==========================================
    with tab5:
        st.header("🎨 Creative Intelligence")
        st.markdown("Upload creative metadata to see aggregated performance by creative type.")
        creative_file = st.file_uploader("Upload Creative CSV (must contain join column)", type=["csv"], key="creative_upload")
        
        if creative_file is not None:
            c_df = pd.read_csv(creative_file)
            st.success(f"Loaded {len(c_df)} creative rows.")
            
            # Require matching column in main df
            possible_joins = list(set(df.columns).intersection(set(c_df.columns)))
            if possible_joins:
                inferred = infer_creative_columns(c_df)
                preferred_join = inferred.get("join") if inferred.get("join") in possible_joins else possible_joins[0]
                join_options = [preferred_join] + [c for c in possible_joins if c != preferred_join]
                join_col = st.selectbox("Join Column", join_options, help="Column that exists in both Main Data and Creative Data")
                creative_group_options = [c for c in c_df.columns if c != join_col]
                default_group = inferred.get("group") if inferred.get("group") in ([join_col] + creative_group_options) else creative_group_options[0] if creative_group_options else join_col
                group_col = st.selectbox(
                    "Creative Grouping",
                    [join_col] + creative_group_options,
                    index=([join_col] + creative_group_options).index(default_group) if default_group in ([join_col] + creative_group_options) else 0,
                    help="Group creatives by hook, angle, format, offer, or creative ID.",
                )
                numeric_cols = [
                    c for c in c_df.columns
                    if pd.to_numeric(c_df[c], errors="coerce").notna().any()
                ]
                outcome_options = numeric_cols or list(c_df.columns)
                inferred_outcome = inferred.get("outcome") if inferred.get("outcome") in outcome_options else outcome_col if outcome_col in outcome_options else outcome_options[0]
                creative_outcome_col = st.selectbox(
                    "Creative Outcome Column",
                    outcome_options,
                    index=outcome_options.index(inferred_outcome),
                    help="Metric to compare by creative group, e.g. results, purchases, revenue, or conversions.",
                )
                treatment_col = st.selectbox(
                    "Treatment Column",
                    ["None"] + list(df.columns),
                    index=(["None"] + list(df.columns)).index(inferred.get("treatment")) if inferred.get("treatment") in df.columns else 0,
                    help="Only select this if the column marks treated vs control rows.",
                )
                period_col = st.selectbox(
                    "Period Column",
                    ["None"] + list(df.columns),
                    index=(["None"] + list(df.columns)).index(inferred.get("period")) if inferred.get("period") in df.columns else 0,
                    help="Only select this if values are pre/post or treatment/control periods.",
                )
                spend_options = [c for c in c_df.columns if c != creative_outcome_col]
                inferred_spend = inferred.get("spend") if inferred.get("spend") in spend_options else "spend" if "spend" in spend_options else spend_options[0] if spend_options else creative_outcome_col
                spend_col = st.selectbox(
                    "Spend Column",
                    spend_options or [creative_outcome_col],
                    index=(spend_options or [creative_outcome_col]).index(inferred_spend),
                )
                
                if st.button("Analyze Creative Lift"):
                    with st.spinner("Aggregating creative performance..."):
                        try:
                            res_df = analyze_creative_lift(
                                df,
                                c_df,
                                join_col=join_col,
                                date_col=date_col,
                                outcome_col=creative_outcome_col,
                                spend_col=spend_col,
                                treatment_col=treatment_col if treatment_col != "None" else "__missing_treatment__",
                                period_col=period_col if period_col != "None" else "__missing_period__",
                                group_col=group_col,
                            )
                            if not res_df.empty:
                                st.dataframe(res_df)
                                if "creative_lift_score" in res_df.columns:
                                    st.bar_chart(res_df.set_index(res_df.columns[0])["creative_lift_score"])
                                if "roas" in res_df.columns:
                                    st.bar_chart(res_df.set_index(res_df.columns[0])["roas"])
                            else:
                                st.warning("Could not aggregate data. Make sure outcome, spend, and creative grouping columns are present.")
                        except Exception as e:
                            st.error(f"Error analyzing creative data: {e}")
            else:
                st.error("No matching columns found between main data and creative metadata.")

    # ==========================================
    # TAB 6: NEXT EXPERIMENT
    # ==========================================
    with tab6:
        st.header("AI Test Recommendation Engine")
        st.markdown("Generate a concrete next experiment plan from the data you uploaded.")

        ne_c1, ne_c2, ne_c3 = st.columns(3)
        expected_lift = ne_c1.number_input("Expected Lift %", min_value=0.01, max_value=1.0, value=0.10, step=0.01, key="ne_lift")
        target_power = ne_c2.number_input("Target Power", min_value=0.5, max_value=0.95, value=0.80, step=0.05, key="ne_power")
        max_duration = ne_c3.number_input("Max Duration", min_value=14, max_value=120, value=60, step=7, key="ne_duration")

        if st.button("Recommend Next Experiment", type="primary", key="ne_button"):
            with st.spinner("Designing next experiment..."):
                plan = recommend_next_experiment(
                    df,
                    date_col=date_col,
                    geo_col=geo_col,
                    outcome_col=outcome_col,
                    input_col=cost_col if cost_col != "None" else None,
                    expected_lift=expected_lift,
                    target_power=target_power,
                    max_duration=int(max_duration),
                )

            if "error" in plan:
                st.error(plan["error"])
            else:
                p1, p2, p3 = st.columns(3)
                p1.metric("Test Geo", plan["test_geo"])
                p2.metric("Duration", f"{plan['duration_days']} days")
                mde = plan.get("minimum_detectable_effect_pct")
                p3.metric("MDE", f"{mde:.0%}" if mde and mde > 0 else "Not reached")

                st.subheader("Experiment Plan")
                st.markdown(f"**Objective:** {plan['objective']}")
                st.markdown(f"**Hypothesis:** {plan['hypothesis']}")
                st.markdown(f"**Primary metric:** {plan['primary_metric']}")
                st.markdown(f"**Success threshold:** {plan['success_threshold']}")
                st.markdown(f"**Interpretation plan:** {plan['interpretation_plan']}")

                st.subheader("Controls and Holdout")
                st.dataframe(pd.DataFrame(plan["control_scores"]), use_container_width=True)

                if plan.get("required_input"):
                    st.subheader("Required Spend / Input Estimate")
                    st.json(plan["required_input"])

                st.subheader("Risks")
                for risk in plan["risks"]:
                    st.warning(risk)

    # ==========================================
    # TAB 7: SCORECARD
    # ==========================================
    with tab7:
        st.header("Incrementality Scorecard")
        registry = ExperimentRegistry()
        scorecard = registry.scorecard()
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Completed", scorecard["completed_experiments"])
        s2.metric("Positive", scorecard["positive_experiments"])
        s3.metric("Positive Rate", f"{scorecard['positive_rate']:.0%}")
        avg_lift = scorecard.get("average_lift_pct")
        s4.metric("Avg Lift", f"{avg_lift:.1%}" if avg_lift is not None else "N/A")

        if scorecard["records"]:
            score_df = pd.DataFrame(scorecard["records"])
            st.dataframe(score_df, use_container_width=True)
            if "lift_pct_mean" in score_df.columns:
                st.line_chart(score_df["lift_pct_mean"])
        else:
            st.info("Run a measurement to start building cumulative incrementality memory.")

else:
    st.info("Please upload a CSV file or connect a data source to begin.")

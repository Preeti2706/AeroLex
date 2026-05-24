"""
streamlit_app.py — AeroLex Demo UI

WHAT:
    Interactive web UI for AeroLex aviation regulatory assistant.
    Connects to FastAPI backend (port 8001) for all queries.

WHY:
    Portfolio demo needs a visual interface — interviewers and
    hiring managers want to SEE the system work, not just read code.
    Streamlit lets us build a professional UI in pure Python —
    no HTML/CSS/JS needed.

HOW:
    1. User selects query type (General/Preflight/Compliance/AD)
    2. User enters query or fills structured form
    3. Streamlit calls FastAPI backend via requests
    4. Response displayed with answer, status, confidence, cost

DESIGN DECISIONS:
    - Sidebar: query type selector + system status
    - Main area: input form + response display
    - Tabs: Answer | Citations | Debug (raw JSON)
    - Color coding: green=approve, yellow=hold, red=block

Official Docs:
    Streamlit: https://docs.streamlit.io/
"""

import time
import requests
import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AeroLex — Aviation Regulatory Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8001"

STATUS_COLORS = {
    "AUTO_APPROVE": "🟢",
    "HOLD":         "🟡",
    "BLOCK":        "🔴",
    "ERROR":        "❌",
}

VERDICT_COLORS = {
    "COMPLIANT":     "✅",
    "NON_COMPLIANT": "❌",
    "UNCLEAR":       "⚠️",
}

# ── Custom CSS ────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F4E8C;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #595959;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #F5F9FC;
        border-left: 4px solid #1F4E8C;
        padding: 1rem 1.5rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
    }
    .status-approve { color: #1E7B34; font-weight: bold; font-size: 1.1rem; }
    .status-hold    { color: #BF8F00; font-weight: bold; font-size: 1.1rem; }
    .status-block   { color: #C00000; font-weight: bold; font-size: 1.1rem; }
    .stAlert > div  { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── API Helpers ───────────────────────────────────────────────────────────

def check_api_health() -> dict:
    """Check if FastAPI backend is running."""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        return resp.json() if resp.status_code == 200 else {}
    except Exception:
        return {}


def call_generic_query(query: str) -> dict:
    """Call POST /api/v1/query."""
    try:
        resp = requests.post(
            f"{API_BASE}/api/v1/query",
            json={"query": query},
            timeout=300,
        )
        return resp.json()
    except Exception as e:
        return {"error": str(e), "status": "ERROR", "answer": "API call failed"}


def call_preflight(
    flight_type: str,
    aircraft_type: str,
    jurisdiction: str,
    specific_question: str = None,
) -> dict:
    """Call POST /api/v1/preflight."""
    try:
        payload = {
            "flight_type":       flight_type,
            "aircraft_type":     aircraft_type,
            "jurisdiction":      jurisdiction,
            "specific_question": specific_question or None,
        }
        resp = requests.post(
            f"{API_BASE}/api/v1/preflight",
            json=payload,
            timeout=300,
        )
        return resp.json()
    except Exception as e:
        return {"error": str(e), "status": "ERROR", "answer": "API call failed"}


def call_compliance(
    scenario: str,
    regulation_part: str = None,
    jurisdiction: str = "FAA",
) -> dict:
    """Call POST /api/v1/compliance."""
    try:
        payload = {
            "scenario":        scenario,
            "regulation_part": regulation_part or None,
            "jurisdiction":    jurisdiction,
        }
        resp = requests.post(
            f"{API_BASE}/api/v1/compliance",
            json=payload,
            timeout=300,
        )
        return resp.json()
    except Exception as e:
        return {"error": str(e), "status": "ERROR", "answer": "API call failed"}


def call_ad_check(aircraft_model: str, query: str = None) -> dict:
    """Call POST /api/v1/ad-check."""
    try:
        payload = {
            "aircraft_model": aircraft_model,
            "query":          query or None,
        }
        resp = requests.post(
            f"{API_BASE}/api/v1/ad-check",
            json=payload,
            timeout=300,
        )
        return resp.json()
    except Exception as e:
        return {"error": str(e), "status": "ERROR", "answer": "API call failed"}


# ── Response Display ──────────────────────────────────────────────────────

def display_response(response: dict, query_type: str):
    """
    Display formatted API response.

    Shows:
    - Status badge (AUTO_APPROVE / HOLD / BLOCK)
    - Answer in styled box
    - Metrics: confidence, cost, latency
    - Compliance verdict (if ADVISORY)
    - Warning (if any)
    - Tabs: Answer | Debug JSON
    """
    if "error" in response and response.get("status") == "ERROR":
        st.error(f"❌ API Error: {response.get('error', 'Unknown error')}")
        return

    status = response.get("status", "UNKNOWN")
    answer = response.get("answer", "No answer returned")

    # ── Status Banner ──
    icon = STATUS_COLORS.get(status, "❓")
    if status == "AUTO_APPROVE":
        st.success(f"{icon} **{status}** — Answer served directly")
    elif status == "HOLD":
        st.warning(f"{icon} **{status}** — Under expert review")
    elif status == "BLOCK":
        st.error(f"{icon} **{status}** — Insufficient regulatory context")
    else:
        st.error(f"{icon} **{status}**")

    # ── Compliance Verdict ──
    verdict = response.get("compliance_verdict")
    if verdict:
        v_icon = VERDICT_COLORS.get(verdict, "❓")
        if verdict == "COMPLIANT":
            st.success(f"{v_icon} **Compliance Verdict: {verdict}**")
        elif verdict == "NON_COMPLIANT":
            st.error(f"{v_icon} **Compliance Verdict: {verdict}**")
        else:
            st.warning(f"{v_icon} **Compliance Verdict: {verdict}** — Consult FSDO/DGCA")

    # ── Tabs ──
    tab1, tab2, tab3 = st.tabs(["📋 Answer", "📊 Metrics", "🔧 Debug"])

    with tab1:
        st.markdown(
            f'<div class="answer-box">{answer}</div>',
            unsafe_allow_html=True
        )

        # Key requirements (preflight)
        key_reqs = response.get("key_requirements", [])
        if key_reqs:
            st.markdown("**Key Requirements:**")
            for req in key_reqs:
                st.markdown(f"• {req}")

        # Reasoning steps (compliance)
        steps = response.get("reasoning_steps", [])
        if steps:
            st.markdown("**Reasoning Steps:**")
            for i, step in enumerate(steps, 1):
                st.markdown(f"**Step {i}:** {step}")

        # Warning
        warning = response.get("warning")
        if warning:
            st.warning(f"⚠️ {warning}")

        # Escalation
        escalation = response.get("escalation_contact")
        if escalation:
            st.info(f"📞 **Escalation Contact:** {escalation}")

    with tab2:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            confidence = response.get("confidence", 0.0)
            st.metric(
                "Confidence",
                f"{confidence:.3f}",
                delta=f"{'✅ Good' if confidence > 0.6 else '⚠️ Low'}"
            )
        with col2:
            cost = response.get("cost_usd", 0.0)
            st.metric("Cost", f"${cost:.6f}")
        with col3:
            latency = response.get("latency_ms", 0.0)
            st.metric("Latency", f"{latency:.0f}ms")
        with col4:
            gate_id = response.get("gate_id", "N/A")
            st.metric("Gate ID", gate_id[:8] if gate_id else "N/A")

        # Query info
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**Query Type:** {response.get('query_type', 'N/A')}")
            st.write(f"**Strategy:** {response.get('strategy', 'N/A')}")
        with col_b:
            query_used = response.get("query_used", response.get("query", "N/A"))
            st.write(f"**Query Used:** {query_used[:100]}...")

    with tab3:
        st.json(response)


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ✈️ AeroLex")
    st.markdown("*Aviation Regulatory Assistant*")
    st.markdown("---")

    # System status
    st.markdown("### 🔌 System Status")
    health = check_api_health()

    if health:
        st.success("✅ API Connected")
        st.write(f"Qdrant: {health.get('qdrant', 'unknown')}")
        st.write(f"MLflow: {health.get('mlflow', 'unknown')}")
        collections = health.get("collections", [])
        st.write(f"Collections: {len(collections)}")
        for c in collections:
            st.write(f"  • {c}")
    else:
        st.error("❌ API Offline — Start FastAPI first")
        st.code("python -m src.api.main")

    st.markdown("---")

    # Query type selector
    st.markdown("### 🎯 Query Mode")
    query_mode = st.radio(
        "Select query type:",
        options=["🔍 General Query", "✈️ Preflight Check",
                 "⚖️ Compliance Check", "🔧 AD Check"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📊 About")
    st.write("**Model:** claude-sonnet-4-5")
    st.write("**Retrieval:** Voyage + BM25 + RRF")
    st.write("**Reranker:** Voyage rerank-2")
    st.write("**Corpus:** 624 Part 91 chunks")
    st.markdown("---")
    st.markdown(
        "📁 [GitHub](https://github.com/Preeti2706/AeroLex) | "
        "📖 [Docs](/redoc)"
    )


# ── Main Area ─────────────────────────────────────────────────────────────

st.markdown(
    '<div class="main-header">✈️ AeroLex</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">Aviation Regulatory Compliance Assistant — '
    'FAA Part 91 · DGCA CARs · Airworthiness Directives</div>',
    unsafe_allow_html=True
)

# ── Query Forms ───────────────────────────────────────────────────────────

if query_mode == "🔍 General Query":
    st.markdown("### Ask Any Aviation Regulatory Question")

    example_queries = [
        "What does 14 CFR 91.103 say about preflight requirements?",
        "How do FAA and DGCA preflight rules differ?",
        "What are the VFR weather minimums under Part 91?",
    ]

    selected_example = st.selectbox(
        "Or pick an example:",
        ["(type your own)"] + example_queries
    )

    query_input = st.text_area(
        "Your question:",
        value=selected_example if selected_example != "(type your own)" else "",
        height=100,
        placeholder="e.g. What must a pilot check before beginning a flight?"
    )

    if st.button("🚀 Ask AeroLex", type="primary", use_container_width=True):
        if not query_input.strip():
            st.warning("Please enter a question")
        else:
            with st.spinner("🔍 Searching regulatory corpus..."):
                response = call_generic_query(query_input.strip())
            display_response(response, "GENERAL")

elif query_mode == "✈️ Preflight Check":
    st.markdown("### Preflight Compliance Check")

    col1, col2, col3 = st.columns(3)
    with col1:
        flight_type = st.selectbox(
            "Flight Type", ["VFR", "IFR", "SVFR"]
        )
    with col2:
        aircraft_type = st.selectbox(
            "Aircraft Type", ["general", "transport", "rotorcraft"]
        )
    with col3:
        jurisdiction = st.selectbox(
            "Jurisdiction", ["FAA", "DGCA"]
        )

    specific_q = st.text_input(
        "Specific question (optional):",
        placeholder="e.g. What fuel reserves are required for VFR day flight?"
    )

    if st.button("✈️ Check Preflight Requirements", type="primary", use_container_width=True):
        with st.spinner("🔍 Checking preflight regulations..."):
            response = call_preflight(
                flight_type, aircraft_type, jurisdiction,
                specific_q if specific_q else None
            )
        display_response(response, "PREFLIGHT")

elif query_mode == "⚖️ Compliance Check":
    st.markdown("### Regulatory Compliance Check")
    st.info(
        "💡 Describe your operational scenario. "
        "AeroLex will provide step-by-step regulatory analysis "
        "and a COMPLIANT / NON-COMPLIANT / UNCLEAR verdict."
    )

    scenario = st.text_area(
        "Describe your scenario:",
        height=120,
        placeholder="e.g. I want to depart with an inoperative altimeter on a VFR day flight under Part 91"
    )

    col1, col2 = st.columns(2)
    with col1:
        reg_part = st.text_input(
            "Regulation Part (optional):",
            placeholder="91"
        )
    with col2:
        jurisdiction = st.selectbox("Jurisdiction", ["FAA", "DGCA"])

    if st.button("⚖️ Check Compliance", type="primary", use_container_width=True):
        if not scenario.strip():
            st.warning("Please describe your scenario")
        else:
            with st.spinner("⚖️ Running compliance analysis..."):
                response = call_compliance(
                    scenario.strip(),
                    reg_part if reg_part else None,
                    jurisdiction,
                )
            display_response(response, "COMPLIANCE")

elif query_mode == "🔧 AD Check":
    st.markdown("### Airworthiness Directive Check")
    st.warning(
        "⚠️ ADs are mandatory. Non-compliance grounds the aircraft. "
        "Always verify with the official FAA AD database."
    )

    aircraft_model = st.text_input(
        "Aircraft Model:",
        placeholder="e.g. Boeing 737-800, Cessna 172S, Airbus A320"
    )

    ad_query = st.text_input(
        "Specific AD question (optional):",
        placeholder="e.g. What ADs apply to the CFM56-7B engine?"
    )

    if st.button("🔧 Check ADs", type="primary", use_container_width=True):
        if not aircraft_model.strip():
            st.warning("Please enter aircraft model")
        else:
            with st.spinner("🔍 Checking airworthiness directives..."):
                response = call_ad_check(
                    aircraft_model.strip(),
                    ad_query if ad_query else None,
                )
            display_response(response, "AD_CHECK")

# ── Footer ────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#595959; font-size:0.85rem;'>"
    "AeroLex v1.0.0 — Portfolio Project | "
    "Preeti | United Airlines | "
    "Built with LangGraph + Qdrant + Claude + FastAPI + Streamlit"
    "</div>",
    unsafe_allow_html=True
)
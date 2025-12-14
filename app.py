import re
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression


# =========================
# Page config + styling
# =========================
st.set_page_config(
    page_title="Jumbo & Company — Device Insurance Attach% Insights",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
/* --- Modern, clean UI --- */
:root{
    --bg:#ffffff;
    --card:#f8f9fa;
    --card2:#f0f2f5;
    --text:#1a1a1a;
    --muted:#4a5568;
    --accent:#0056b3;
    --accent-alt:#204492;
    --accent-cool:#1db5e9;
    --good:#0f9d58;
    --bad:#d64545;
    --warn:#f26f2c;
    --border:rgba(0,0,0,0.14);
    --shadow: 0 12px 32px rgba(0,0,0,0.12);
  --radius:18px;
}
.main > div {padding-top: 2.1rem; background: var(--bg);} 
.block-container{max-width: 1350px; padding-top: 2.1rem; background: var(--bg);} 
[data-testid="stSidebar"] {background: linear-gradient(180deg, #f8f9fa 0%, #eef1f6 100%); border-right: 1px solid var(--border);} 
h1, h2, h3, h4, h5, h6, p, span, div {color: var(--text) !important;}
.small-muted{color: var(--muted) !important; font-size: 0.9rem;}
.badge{
    display:inline-block; padding: 0.25rem 0.6rem; border-radius: 999px;
    background: linear-gradient(120deg, rgba(0,86,179,0.12), rgba(29,181,233,0.12));
    border: 1px solid rgba(0,86,179,0.20);
    color: var(--text); font-size: 0.85rem; margin-right: 0.35rem;
}
.kpi{
    background: linear-gradient(145deg, rgba(255,255,255,0.85), rgba(240,242,245,0.9));
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.05rem 1.05rem;
    box-shadow: var(--shadow);
}
.kpi .label{color: var(--muted) !important; font-size: 0.9rem;}
.kpi .value{font-size: 1.65rem; font-weight: 700; color: var(--accent) !important;}
.kpi .delta{margin-top: 0.15rem; font-size: 0.95rem; color: var(--muted) !important;}
.panel{
    background: linear-gradient(145deg, rgba(248,249,250,0.95), rgba(240,242,245,0.95));
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.1rem;
    box-shadow: var(--shadow);
}
.hr{height:1px;background:var(--border);margin:0.9rem 0;}
.stButton > button{
    border-radius: 14px;
    border: 1px solid rgba(0,86,179,0.35);
    background: rgba(0,86,179,0.12);
    color: var(--text);
    padding: 0.5rem 0.9rem;
}
.stButton > button:hover{border-color: rgba(0,86,179,0.65); background: rgba(0,86,179,0.18);}
a{color: var(--accent) !important;}

.brand-row{
    display:flex; align-items:center; gap:0.75rem; margin:1.0rem 0 1.1rem 0;
    padding:0.4rem 0.75rem; border:1px solid var(--border); border-radius:14px;
    background: linear-gradient(120deg, rgba(0,86,179,0.08), rgba(29,181,233,0.08));
}
.brand-row img{height:42px;}
.brand-name{font-weight:700; color: var(--accent-alt); font-size:1.05rem;}
.brand-tag{color: var(--muted); font-size:0.95rem;}

/* Enhanced styles for better readability */
.stDataFrame {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

.stDataFrame table {
    background: var(--card2) !important;
}

.stDataFrame th {
    background: rgba(0,86,179,0.08) !important;
    font-weight: 600 !important;
    border-bottom: 1px solid var(--border) !important;
}

.stDataFrame tr:hover {
    background: rgba(0,86,179,0.06) !important;
}

/* Better form controls */
.stSelectbox, .stMultiselect, .stSlider {
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Better expander */
.streamlit-expanderHeader {
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Breadcrumb styling */
.breadcrumb {
    background: rgba(0,86,179,0.06); 
    padding: 0.5rem 1rem; 
    border-radius: 10px; 
    margin-bottom: 1.5rem; 
    border: 1px solid var(--border);
}

/* Section header */
.section-header {
    margin-bottom: 1.5rem;
}

.section-header h3 {
    color: var(--text); 
    margin-bottom: 0.25rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Logo
LOGO_PATH = Path(__file__).parent / "data" / "zopper_logo.svg"
LOGO_B64 = base64.b64encode(LOGO_PATH.read_bytes()).decode() if LOGO_PATH.exists() else None


# =========================
# Helpers
# =========================
MONTHS_ORDER = ["Aug", "Sep", "Oct", "Nov", "Dec"]  # as per the provided sheet
MONTH_TO_IDX = {m: i + 1 for i, m in enumerate(MONTHS_ORDER)}  # Aug=1,...Dec=5
IDX_TO_MONTH = {v: k for k, v in MONTH_TO_IDX.items()}

def _coerce_pct(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace("%", "")
    try:
        v = float(s)
    except Exception:
        return np.nan
    # if someone typed 23 instead of 0.23, normalize
    if v > 1.5:
        v = v / 100.0
    return v

def load_data(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Basic expected cols: Branch, Store_Name, Aug..Dec (any subset ok)
    if "Branch" not in df.columns:
        raise ValueError("Missing column: Branch")
    store_col = "Store_Name" if "Store_Name" in df.columns else ("Store" if "Store" in df.columns else None)
    if store_col is None:
        raise ValueError("Missing store column (Store_Name/Store).")
    df = df.rename(columns={store_col: "Store"})
    # keep only known months present
    month_cols = [m for m in MONTHS_ORDER if m in df.columns]
    if not month_cols:
        raise ValueError("No month columns found (Aug..Dec).")
    for m in month_cols:
        df[m] = df[m].apply(_coerce_pct)

    df = df[["Branch", "Store"] + month_cols].dropna(subset=["Branch", "Store"])
    df["Branch"] = df["Branch"].astype(str).str.strip()
    df["Store"] = df["Store"].astype(str).str.strip()

    return df, month_cols

def to_long(df_wide: pd.DataFrame, month_cols):
    long = df_wide.melt(id_vars=["Branch", "Store"], value_vars=month_cols,
                        var_name="Month", value_name="AttachPct")
    long["Month"] = long["Month"].astype(str).str.strip()
    long["t"] = long["Month"].map(MONTH_TO_IDX)
    long = long.dropna(subset=["AttachPct", "t"]).sort_values(["Branch", "Store", "t"])
    return long

def add_store_metrics(long: pd.DataFrame):
    g = long.groupby(["Branch", "Store"])
    metrics = g["AttachPct"].agg(avg="mean", min="min", max="max", std="std").reset_index()
    metrics["std"] = metrics["std"].fillna(0)
    # Trend via slope of linear regression per store (AttachPct ~ t)
    slopes = []
    last_vals = []
    for (b, s), d in g:
        X = d[["t"]].values
        y = d["AttachPct"].values
        if len(d) >= 2:
            lr = LinearRegression().fit(X, y)
            slope = float(lr.coef_[0])
        else:
            slope = 0.0
        slopes.append(((b, s), slope))
        last_vals.append(((b, s), float(d.sort_values("t")["AttachPct"].iloc[-1])))
    slope_df = pd.DataFrame([{"Branch": k[0], "Store": k[1], "slope": v} for k, v in slopes])
    last_df  = pd.DataFrame([{"Branch": k[0], "Store": k[1], "last_attach": v} for k, v in last_vals])
    out = metrics.merge(slope_df, on=["Branch","Store"], how="left").merge(last_df, on=["Branch","Store"], how="left")
    out["volatility"] = out["std"]
    return out

def store_segmentation(store_metrics: pd.DataFrame, n_clusters=4):
    X = store_metrics[["avg", "volatility", "slope"]].copy()
    # scale with robust-ish normalization
    X = (X - X.median()) / (X.quantile(0.75) - X.quantile(0.25) + 1e-9)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X.values)
    seg = store_metrics.copy()
    seg["segment_id"] = labels

    # segment naming based on centroid profile in original space
    cent = seg.groupby("segment_id")[["avg","volatility","slope"]].mean()
    names = {}
    for sid, row in cent.iterrows():
        if row["avg"] >= cent["avg"].quantile(0.75) and row["slope"] >= 0:
            names[sid] = "Champions (High & improving)"
        elif row["avg"] >= cent["avg"].median() and row["slope"] < 0:
            names[sid] = "At-risk (High but falling)"
        elif row["avg"] < cent["avg"].median() and row["slope"] > 0:
            names[sid] = "Risers (Low but improving)"
        else:
            names[sid] = "Long tail (Low & flat)"
    seg["segment"] = seg["segment_id"].map(names).fillna("Segment")
    return seg

def predict_jan(store_metrics: pd.DataFrame, long: pd.DataFrame):
    """
    Forecast Jan (t=6) per store using a simple, explainable ensemble:
      - per-store linear trend extrapolation (Aug..Dec => t=1..5, predict t=6)
      - branch-level trend as fallback / regularization
    """
    g_store = long.groupby(["Branch", "Store"])
    g_branch = long.groupby(["Branch"])

    # branch model: AttachPct ~ t
    branch_pred = {}
    for b, d in g_branch:
        X = d[["t"]].values
        y = d["AttachPct"].values
        if len(d) >= 2:
            lr = LinearRegression().fit(X, y)
            pred = float(lr.predict(np.array([[6]]) )[0])
        else:
            pred = float(np.nanmean(y)) if len(d) else 0.0
        branch_pred[b] = pred

    rows = []
    for (b, s), d in g_store:
        X = d[["t"]].values
        y = d["AttachPct"].values
        if len(d) >= 2:
            lr = LinearRegression().fit(X, y)
            store_jan = float(lr.predict(np.array([[6]]) )[0])
        else:
            store_jan = float(y[-1]) if len(d) else np.nan

        br_jan = float(branch_pred.get(b, np.nan))
        # regularize: more volatility => more weight on branch
        vol = float(store_metrics.loc[(store_metrics.Branch==b)&(store_metrics.Store==s), "volatility"].iloc[0])
        w_branch = np.clip(vol / 0.12, 0.15, 0.65)  # tuned to typical attach% scale
        w_store = 1.0 - w_branch
        pred = w_store * store_jan + w_branch * br_jan

        # soft clipping to plausible bounds
        pred = float(np.clip(pred, 0.0, 1.0))

        # confidence heuristic: lower volatility + more points = higher confidence
        n = len(d)
        conf = float(np.clip(1.0 - (vol / 0.18), 0.0, 1.0)) * (0.7 + 0.3 * min(1.0, n/5))
        rows.append({
            "Branch": b,
            "Store": s,
            "Jan_Pred_AttachPct": pred,
            "Model_Confidence": conf,
            "Store_Trend_Slope": float(store_metrics.loc[(store_metrics.Branch==b)&(store_metrics.Store==s), "slope"].iloc[0]),
            "Volatility": vol,
            "Last_Month_AttachPct": float(d.sort_values("t")["AttachPct"].iloc[-1]),
        })
    pred_df = pd.DataFrame(rows)
    return pred_df

def fmt_pct(x):
    if pd.isna(x):
        return "—"
    return f"{x*100:.1f}%"

def kpi(label, value, delta=None, help_text=None):
    help_icon = f"<span title='{help_text}' style='margin-left: 0.3rem; color: var(--muted); cursor: help;'>ⓘ</span>" if help_text else ""
    d_html = f"<div class='delta'>{delta}</div>" if delta is not None else ""
    st.markdown(
        f"""
        <div class="kpi">
          <div class="label">{label}{help_icon}</div>
          <div class="value">{value}</div>
          {d_html}
        </div>
        """,
        unsafe_allow_html=True
    )

def section_header(title, description=None, icon=""):
    html = f"""
    <div class='section-header'>
        <h3 style='color: var(--text); margin-bottom: 0.25rem;'>
            {title}
        </h3>
        {f"<p class='small-muted' style='margin-top: 0;'>{description}</p>" if description else ""}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def tooltip(text, icon="ⓘ"):
    return f'<span title="{text}" style="color: var(--muted); cursor: help; margin-left: 0.3rem;">{icon}</span>'

def styled_dataframe(df, height=300):
    return st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=height,
        column_config={
            col: st.column_config.Column(
                help=f"Column: {col}"
            ) for col in df.columns
        }
    )


# =========================
# Sidebar
# =========================
if LOGO_B64:
    st.sidebar.image(f"data:image/svg+xml;base64,{LOGO_B64}", use_container_width=True)
    st.sidebar.markdown("<div class='hr'></div>", unsafe_allow_html=True)

st.sidebar.markdown("## Controls")
st.sidebar.markdown("<div class='small-muted'>Upload the given sheet or use the bundled sample (converted from .xls).</div>", unsafe_allow_html=True)

default_path = Path(__file__).parent / "data" / "Jumbo_Attach_Sample.xlsx"
use_sample = st.sidebar.toggle("Use sample file", value=True)

uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"]) if not use_sample else None

st.sidebar.markdown("<div class='hr'></div>", unsafe_allow_html=True)
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "",
    ["Executive Summary", 
     "Branch & Month Insights", 
     "Store Deep Dive", 
     "Store Segments", 
     "Forecast: January", 
     "Download Pack"],
    index=0,
    label_visibility="collapsed"
)

st.sidebar.markdown("<div class='hr'></div>", unsafe_allow_html=True)
st.sidebar.markdown("### Data Controls")
st.sidebar.markdown("<div class='small-muted'>Upload your data file or use the provided sample dataset.</div>", unsafe_allow_html=True)

st.sidebar.markdown("<div class='hr'></div>", unsafe_allow_html=True)
st.sidebar.markdown("### What this app produces")
st.sidebar.markdown(
    """
- **Leaderboard** of top/bottom stores & branches
- **Month-over-month** movement and volatility
- **Smart store** categorization (segments)
- **Explainable Jan forecast** per store (with confidence)
- **One-click downloads** (clean data + forecast)
""",
)


# =========================
# Load
# =========================
@st.cache_data(show_spinner=False)
def cached_load(file_path_or_bytes):
    return load_data(file_path_or_bytes)

@st.cache_data(show_spinner=False)
def cached_long(df_wide, month_cols):
    return to_long(df_wide, month_cols)

@st.cache_data(show_spinner=False)
def cached_metrics(long):
    return add_store_metrics(long)

@st.cache_data(show_spinner=False)
def cached_segments(metrics):
    return store_segmentation(metrics, n_clusters=4)

@st.cache_data(show_spinner=False)
def cached_pred(metrics, long):
    return predict_jan(metrics, long)

def get_df():
    if use_sample:
        if not default_path.exists():
            raise FileNotFoundError("Sample file missing: app/data/Jumbo_Attach_Sample.xlsx")
        return cached_load(default_path)
    if uploaded is None:
        return None
    return cached_load(uploaded)

def load_with_progress():
    with st.spinner("Loading and processing data..."):
        loaded = get_df()
        if loaded is None:
            st.info("Upload an .xlsx file to begin, or toggle **Use sample file**.")
            st.stop()
        return loaded

# Load data with progress indicator (non-cutting banner)
df_wide, month_cols = load_with_progress()
long = cached_long(df_wide, month_cols)
store_metrics = cached_metrics(long)
segments = cached_segments(store_metrics)
pred = cached_pred(store_metrics, long).merge(segments[["Branch","Store","segment"]], on=["Branch","Store"], how="left")


# =========================
# Header
# =========================
if LOGO_B64:
    st.markdown(
        f"""
        <div class='brand-row'>
            <img src="data:image/svg+xml;base64,{LOGO_B64}" alt="Zopper logo" />
            <div>
                <div class='brand-name'>Zopper</div>
                <div class='brand-tag'>Device Insurance Attach% Insights</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("## 📊 Jumbo & Company — Device Insurance Attach% Analytics")
st.markdown(
    "<p class='small-muted' style='margin-top: -0.5rem; margin-bottom: 1.5rem;'>Interactive dashboard for store performance analysis, segmentation, and forecasting</p>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.5rem;'>
        <span class='badge'>📈 Executive Insights</span>
        <span class='badge'>🏪 Branch & Store Analysis</span>
        <span class='badge'>🎯 Store Segmentation</span>
        <span class='badge'>🔮 January Forecast</span>
        <span class='badge'>📥 Export Ready</span>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# Global KPIs
# =========================
c1, c2, c3, c4 = st.columns(4)
overall_avg = float(long["AttachPct"].mean())
mo_last = long[long["Month"] == month_cols[-1]]["AttachPct"].mean()
mo_first = long[long["Month"] == month_cols[0]]["AttachPct"].mean()
delta = mo_last - mo_first

with c1:
    kpi("Overall Avg Attach%", 
        fmt_pct(overall_avg), 
        f"Aug→Dec change: {fmt_pct(delta)}",
        "Average attach rate across all stores and months")
with c2:
    kpi("Stores Covered", 
        f"{long['Store'].nunique():,}",
        help_text="Total number of unique stores in dataset")
with c3:
    kpi("Branches Covered", 
        f"{long['Branch'].nunique():,}",
        help_text="Total number of unique branches")
with c4:
    best_store = store_metrics.sort_values("avg", ascending=False).iloc[0]
    kpi("Best Performing Store", 
        f"{best_store['Store']}", 
        f"{best_store['Branch']} • Avg {fmt_pct(best_store['avg'])}",
        "Store with highest average attach rate")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


# =========================
# Pages
# =========================
if page == "Executive Summary":
    # Add breadcrumb
    st.markdown(
        """
        <div class='breadcrumb'>
            <span class='small-muted'>Navigation:</span> 
            <strong>Executive Summary</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        section_header("What's Happening", "Key insights and branch performance at a glance")
        st.markdown(
            """
            This dashboard is built to help a business reader quickly answer:
            - **Which branches and stores** are driving attach%?
            - **Where are we losing** attach% (declining trends / volatile stores)?
            - **Which store cohorts** need different actions?
            - **What is the likely attach%** for **January** store-by-store?
            """
        )

        # Branch leaderboard
        branch_tbl = long.groupby("Branch")["AttachPct"].mean().reset_index().sort_values("AttachPct", ascending=False)
        branch_tbl["AttachPct"] = branch_tbl["AttachPct"].apply(fmt_pct)
        branch_tbl.columns = ["Branch", "Avg Attach%"]
        section_header("Branch Leaderboard", "Top performing branches by average attach rate")
        styled_dataframe(branch_tbl, height=250)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        section_header("Quick Wins & Watchouts", "Stores showing significant improvement or decline")
        
        # Top risers and fallers by slope
        tmp = store_metrics.copy()
        tmp["avg_fmt"] = tmp["avg"].apply(fmt_pct)
        tmp["slope_fmt"] = tmp["slope"].apply(lambda x: f"{x*100:+.2f} pp / month")
        risers = tmp.sort_values("slope", ascending=False).head(7)[["Branch","Store","avg_fmt","slope_fmt"]]
        risers.columns = ["Branch", "Store", "Avg Attach%", "Monthly Trend"]
        
        fallers = tmp.sort_values("slope", ascending=True).head(7)[["Branch","Store","avg_fmt","slope_fmt"]]
        fallers.columns = ["Branch", "Store", "Avg Attach%", "Monthly Trend"]

        section_header("Risers (Improving Stores)")
        styled_dataframe(risers, height=200)
        
        section_header("Watchouts (Declining Stores)")
        styled_dataframe(fallers, height=200)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    section_header("Month-over-month Distribution", "Box plot showing attach rate distribution by month")
    if px:
        fig = px.box(long, x="Month", y="AttachPct", points="all")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(
            margin=dict(l=10,r=10,t=40,b=10), 
            height=420,
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#1a1a1a')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Plotly not available in this environment. Install plotly for interactive charts.")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Branch & Month Insights":
    # Add breadcrumb
    st.markdown(
        """
        <div class='breadcrumb'>
            <span class='small-muted'>Navigation:</span> 
            <a href='#' onclick='window.location.reload()'>Home</a> 
            <span class='small-muted'>→</span> 
            <strong>Branch & Month Insights</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    section_header("Branch Performance by Month", "Analyze monthly trends and performance across branches")
    
    with st.expander("🔍 Filter Options", expanded=False):
        bcol1, bcol2, bcol3 = st.columns([1,1,1])
        branches = sorted(long["Branch"].unique().tolist())
        sel_branches = bcol1.multiselect("Filter branches", branches, default=branches[:3])
        sel_months = bcol2.multiselect("Filter months", month_cols, default=month_cols)
        view = bcol3.selectbox("View", ["Trend line", "Heatmap table"], index=0)

    sub = long[long["Branch"].isin(sel_branches) & long["Month"].isin(sel_months)].copy()

    if view == "Trend line":
        if px:
            br_m = sub.groupby(["Branch","Month"])["AttachPct"].mean().reset_index()
            br_m["Month"] = pd.Categorical(br_m["Month"], categories=MONTHS_ORDER, ordered=True)
            fig = px.line(br_m.sort_values("Month"), x="Month", y="AttachPct", color="Branch", markers=True)
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(
                height=460, 
                margin=dict(l=10,r=10,t=40,b=10),
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(color='#1a1a1a')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            styled_dataframe(sub.head(30))
    else:
        pivot = sub.pivot_table(index="Branch", columns="Month", values="AttachPct", aggfunc="mean")
        pivot = pivot.reindex(columns=MONTHS_ORDER).round(4)
        st.dataframe(pivot.style.format("{:.1%}"), use_container_width=True, height=400)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    section_header("Branch Health Analysis", "Actionable insights by branch performance", "🏥")
    br = long.groupby("Branch")["AttachPct"].agg(avg="mean", std="std").reset_index()
    br["std"] = br["std"].fillna(0)
    br["Health"] = np.where(br["avg"] >= br["avg"].quantile(0.75), "Strong 🟢",
                     np.where(br["avg"] >= br["avg"].median(), "Stable 🟡", "Needs Focus 🔴"))
    br = br.sort_values(["Health","avg"], ascending=[True, False])
    show = br.copy()
    show["Avg Attach%"] = show["avg"].apply(fmt_pct)
    show["Volatility"] = show["std"].apply(lambda x: f"{x*100:.1f} pp")
    show["Health"] = show["Health"]
    styled_dataframe(show[["Branch", "Avg Attach%", "Volatility", "Health"]], height=300)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Store Deep Dive":
    # Add breadcrumb
    st.markdown(
        """
        <div class='breadcrumb'>
            <span class='small-muted'>Navigation:</span> 
            <a href='#' onclick='window.location.reload()'>Home</a> 
            <span class='small-muted'>→</span> 
            <strong>Store Deep Dive</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    section_header("Store Explorer", "Drill down into individual store performance")
    
    with st.expander("Filter & View Options", expanded=True):
        cA, cB, cC = st.columns([1.1, 1.2, 1.0])
        branches = sorted(long["Branch"].unique().tolist())
        sel_branch = cA.selectbox("Branch", ["All"] + branches, index=0)

        stores = sorted(long[long["Branch"].eq(sel_branch)]["Store"].unique().tolist()) if sel_branch != "All" else sorted(long["Store"].unique().tolist())
        sel_store = cB.selectbox("Store", stores, index=0)

        metric_view = cC.selectbox("Focus View", ["Trend", "Compare to branch", "Volatility"], index=0)

    sub = long[(long["Store"] == sel_store)].copy()
    store_row = store_metrics[store_metrics["Store"] == sel_store].iloc[0]
    
    section_header(f"{sel_store}", f"Branch: {store_row['Branch']}")
    
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Avg Attach%", fmt_pct(store_row["avg"]), help="Average attach rate across all months")
    s2.metric("Min→Max Range", f"{fmt_pct(store_row['min'])} → {fmt_pct(store_row['max'])}", help="Performance range across months")
    s3.metric("Monthly Trend", f"{store_row['slope']*100:+.2f} pp / month", help="Positive = improving, Negative = declining")
    s4.metric("Volatility", f"{store_row['volatility']*100:.1f} pp", help="Standard deviation of attach rates")

    if px:
        if metric_view == "Trend":
            sub["Month"] = pd.Categorical(sub["Month"], categories=MONTHS_ORDER, ordered=True)
            fig = px.line(sub.sort_values("Month"), x="Month", y="AttachPct", markers=True)
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(
                height=420, 
                margin=dict(l=10,r=10,t=40,b=10),
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(color='#1a1a1a')
            )
            st.plotly_chart(fig, use_container_width=True)

        elif metric_view == "Compare to branch":
            b = store_row["Branch"]
            br_series = long[long["Branch"] == b].groupby("Month")["AttachPct"].mean().reset_index()
            br_series["Series"] = f"{b} average"
            st_series = sub.groupby("Month")["AttachPct"].mean().reset_index()
            st_series["Series"] = sel_store
            both = pd.concat([br_series, st_series], ignore_index=True)
            both["Month"] = pd.Categorical(both["Month"], categories=MONTHS_ORDER, ordered=True)
            fig = px.line(both.sort_values("Month"), x="Month", y="AttachPct", color="Series", markers=True)
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(
                height=420, 
                margin=dict(l=10,r=10,t=40,b=10),
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(color='#1a1a1a')
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            # volatility view: bar of month-to-month deltas
            sub = sub.sort_values("t")
            sub["MoM Change"] = sub["AttachPct"].diff()
            fig = px.bar(sub, x="Month", y="MoM Change")
            fig.update_yaxes(tickformat="+.0%")
            fig.update_layout(
                height=420, 
                margin=dict(l=10,r=10,t=40,b=10),
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(color='#1a1a1a')
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        styled_dataframe(sub)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    section_header("Store Leaderboard", "Rank stores by different performance metrics")
    
    with st.expander("Ranking Options", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            topn = st.slider("Show top/bottom N stores", 5, 30, 12)
        with col2:
            by = st.selectbox("Rank by", ["Average Attach%", "Trend", "Volatility"], 
                             index=0, format_func=lambda x: {
                                 "Average Attach%": "avg",
                                 "Trend": "slope", 
                                 "Volatility": "volatility"
                             }[x])
    
    # Map display names to column names
    by_col = {"Average Attach%": "avg", "Trend": "slope", "Volatility": "volatility"}[by]
    
    tbl = store_metrics.merge(segments[["Branch","Store","segment"]], on=["Branch","Store"], how="left")
    tbl = tbl.sort_values(by_col, ascending=False)

    show = tbl[["Branch","Store","segment","avg","slope","volatility","last_attach"]].copy()
    show["Avg Attach%"] = show["avg"].apply(fmt_pct)
    show["Last Month"] = show["last_attach"].apply(fmt_pct)
    show["Monthly Trend"] = show["slope"].apply(lambda x: f"{x*100:+.2f} pp/m")
    show["Volatility"] = show["volatility"].apply(lambda x: f"{x*100:.1f} pp")
    
    styled_dataframe(show[["Branch", "Store", "segment", "Avg Attach%", "Monthly Trend", "Volatility", "Last Month"]].head(topn), height=350)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Store Segments":
    # Add breadcrumb
    st.markdown(
        """
        <div class='breadcrumb'>
            <span class='small-muted'>Navigation:</span> 
            <a href='#' onclick='window.location.reload()'>Home</a> 
            <span class='small-muted'>→</span> 
            <strong>Store Segments</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    section_header("Store Categorization (Segments)", "Group stores by performance characteristics for targeted actions")
    st.markdown(
        """
        Stores are grouped using **K-Means clustering** on three explainable features:
        1. **Average attach%** - Overall performance level
        2. **Volatility** - Consistency of performance
        3. **Trend slope** - Direction of performance
        
        This turns hundreds of stores into 3–5 action-oriented cohorts.
        """
    )

    seg_counts = segments["segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Number of Stores"]
    if px:
        fig = px.bar(seg_counts, x="Segment", y="Number of Stores")
        fig.update_layout(
            height=360, 
            margin=dict(l=10,r=10,t=40,b=10),
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#1a1a1a')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    section_header("Segment Distribution", "Number of stores in each segment")
    styled_dataframe(seg_counts, height=150)

    section_header("Segment Playbook (Recommended Actions)", "What to do for each store segment")
    st.markdown(
        """
                - **Champions (High & improving):** Replicate scripts, incentives, and promoter staff. 
          *Best practice sharing opportunities.*
          
                - **At-risk (High but falling):** Investigate churn in sales staff, stock-outs, counter practices. 
          *Refresh pitch and retrain staff.*
          
                - **Risers (Low but improving):** Double down on training and nudges. 
          *Best ROI cohort for incremental investment.*
          
                - **Long tail (Low & flat):** Consider targeted interventions (bundles), or reduce effort and focus on big wins. 
          *Evaluate store viability.*
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    section_header("Segment Details", "View all stores with their segment assignments")
    
    seg = segments.copy()
    seg["Avg Attach%"] = seg["avg"].apply(fmt_pct)
    seg["Trend"] = seg["slope"].apply(lambda x: f"{x*100:+.2f} pp/m")
    seg["Volatility"] = seg["volatility"].apply(lambda x: f"{x*100:.1f} pp")
    seg["Last Month"] = seg["last_attach"].apply(fmt_pct)
    
    styled_dataframe(seg[["Branch","Store","segment","Avg Attach%","Trend","Volatility","Last Month"]], height=400)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Forecast: January":
    # Add breadcrumb
    st.markdown(
        """
        <div class='breadcrumb'>
            <span class='small-muted'>Navigation:</span> 
            <a href='#' onclick='window.location.reload()'>Home</a> 
            <span class='small-muted'>→</span> 
            <strong>Forecast: January</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    section_header("January Attach% Prediction (Store-level)", "Forecast using trend analysis and branch regularization")
    st.markdown(
        """
        **Forecast methodology:**
        - Uses **trend extrapolation** per store (Aug-Dec → Jan)
        - **Regularized by branch trend** for stability
        - **Confidence score** based on volatility and data points
        - **Simple and explainable** for business discussions
        """
    )

    with st.expander("Filter & Sort Options", expanded=False):
        f1, f2, f3 = st.columns([1.1,1.1,1.0])
        branches = sorted(pred["Branch"].unique().tolist())
        sel_br = f1.selectbox("Branch filter", ["All"] + branches, index=0)
        segs = sorted(pred["segment"].dropna().unique().tolist())
        sel_seg = f2.selectbox("Segment filter", ["All"] + segs, index=0)
        sort_by_display = f3.selectbox("Sort by", 
                                     ["Jan Prediction", "Model Confidence", "Store Trend", "Volatility"], 
                                     index=0)
        
        # Map display names to column names
        sort_by_map = {
            "Jan Prediction": "Jan_Pred_AttachPct",
            "Model Confidence": "Model_Confidence",
            "Store Trend": "Store_Trend_Slope",
            "Volatility": "Volatility"
        }
        sort_by = sort_by_map[sort_by_display]

    sub = pred.copy()
    if sel_br != "All":
        sub = sub[sub["Branch"] == sel_br]
    if sel_seg != "All":
        sub = sub[sub["segment"] == sel_seg]

    sub = sub.sort_values(sort_by, ascending=False)

    view = sub[["Branch","Store","segment","Jan_Pred_AttachPct","Model_Confidence","Last_Month_AttachPct","Store_Trend_Slope","Volatility"]].copy()
    view["Jan Prediction"] = view["Jan_Pred_AttachPct"].apply(fmt_pct)
    view["Last Month"] = view["Last_Month_AttachPct"].apply(fmt_pct)
    view["Confidence"] = view["Model_Confidence"].apply(lambda x: f"{x*100:.0f}%")
    view["Trend"] = view["Store_Trend_Slope"].apply(lambda x: f"{x*100:+.2f} pp/m")
    view["Volatility"] = view["Volatility"].apply(lambda x: f"{x*100:.1f} pp")
    
    styled_dataframe(view[["Branch","Store","segment","Jan Prediction","Confidence","Last Month","Trend","Volatility"]], height=400)

    if px:
        section_header("Top Predictions Visualization", "Bar chart of top predicted stores")
        topk = st.slider("Number of stores to plot", 10, 50, 20, key="topk_slider")
        plot_df = sub.head(topk).copy()
        plot_df["Jan_Pred_AttachPct_pct"] = plot_df["Jan_Pred_AttachPct"] * 100
        fig = px.bar(plot_df, x="Store", y="Jan_Pred_AttachPct_pct", color="Branch")
        fig.update_layout(
            height=460, 
            margin=dict(l=10,r=10,t=40,b=10),
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#1a1a1a')
        )
        fig.update_yaxes(title="Jan predicted attach%")
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Download Pack":
    # Add breadcrumb
    st.markdown(
        """
        <div class='breadcrumb'>
            <span class='small-muted'>Navigation:</span> 
            <a href='#' onclick='window.location.reload()'>Home</a> 
            <span class='small-muted'>→</span> 
            <strong>Download Pack</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    section_header("Data Export", "Download cleaned data, metrics, segments, and forecasts")
    st.markdown("<div class='small-muted'>Export the analysis results for reporting or further analysis.</div>", unsafe_allow_html=True)

    # Prepare data for export
    long_export = long.copy()
    long_export["AttachPct"] = long_export["AttachPct"].round(6)

    metrics_export = segments.merge(pred[["Branch","Store","Jan_Pred_AttachPct","Model_Confidence"]], on=["Branch","Store"], how="left")
    metrics_export = metrics_export.rename(columns={"Jan_Pred_AttachPct":"Jan_Pred"})

    def to_bytes(df):
        return df.to_csv(index=False).encode("utf-8")

    # Individual CSV downloads
    section_header("Individual CSV Downloads", "Download specific datasets as CSV files")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            "⬇️ Long Format Data", 
            data=to_bytes(long_export), 
            file_name="attach_long_format.csv", 
            mime="text/csv",
            help="Month-by-month attach rates for all stores"
        )
    
    with col2:
        st.download_button(
            "⬇️ Store Metrics & Segments", 
            data=to_bytes(metrics_export), 
            file_name="store_segments_metrics.csv", 
            mime="text/csv",
            help="Store-level metrics with segment assignments"
        )
    
    with col3:
        st.download_button(
            "⬇️ January Forecast", 
            data=to_bytes(pred), 
            file_name="january_forecast.csv", 
            mime="text/csv",
            help="January predictions with confidence scores"
        )

    # Excel workbook export
    section_header("Complete Excel Workbook", "Single file with all analysis results")
    st.markdown("<div class='small-muted'>Comprehensive Excel file with multiple sheets for easy sharing.</div>", unsafe_allow_html=True)
    
    try:
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_wide.to_excel(writer, index=False, sheet_name="01_Raw_Data")
            long_export.to_excel(writer, index=False, sheet_name="02_Long_Format")
            store_metrics.to_excel(writer, index=False, sheet_name="03_Store_Metrics")
            segments.to_excel(writer, index=False, sheet_name="04_Segments")
            pred.to_excel(writer, index=False, sheet_name="05_January_Forecast")
        
        st.download_button(
            "⬇️ Download Complete Excel Pack", 
            data=output.getvalue(), 
            file_name="attach_insights_pack.xlsx",
            help="All data in a single Excel file with multiple sheets"
        )
    except Exception as e:
        st.warning("Excel export requires openpyxl. Install with: `pip install openpyxl`")
        st.info("Using sample data? CSV downloads are still available.")

    st.markdown("</div>", unsafe_allow_html=True)

import re
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "poll.csv"
DATAMAP_PATH = BASE_DIR / "data" / "datamap.xlsx"



st.set_page_config(page_title="Media Habits 2026 Poll Explorer", layout="wide")
st.title("Media Habits 2026 Poll Explorer")
st.caption("If you have any questions, Slack Alex.")


# -----------------------------
# Filters (dimensions)
# -----------------------------
FILTER_VARS = {
    "DC Metro Area? (Q_DC)": "Q_DC",
    "Party (Q1B_PARTY_CODE)": "Q1B_PARTY_CODE",
    "Occupation (CODE_OCCUPATION)": "CODE_OCCUPATION",
    "Race / Ethnicity (Q_RACE)": "Q_RACE",
    "Income (Q_INCOME)": "Q_INCOME",
}

AGE_VAR = "Q_AGE"  # birth year -> we'll derive age


# -----------------------------
# Loaders
# -----------------------------
@st.cache_data(show_spinner=False)
def load_poll(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def load_datamap(path: str) -> pd.DataFrame:
    dm = pd.read_excel(path, sheet_name="Datamap")
    dm.columns = [str(c).strip() for c in dm.columns]
    return dm

def _s(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


# -----------------------------
# Datamap parsing (fits your file)
# -----------------------------
@st.cache_data(show_spinner=False)
def parse_datamap(dm: pd.DataFrame):
    """
    Robust parse:
      - qtext comes from header rows: "[Q_X]: Question text"
      - single_maps from numeric-ish codes in col1 + label in col2
      - multi_opts from bracketed option columns in col1 like "[Q_NEWS_LASTr12]"
        -> base var derived from option col name prefix BEFORE the 'r#'
      - EXCLUDES open-ended option columns like r95oe from multi_opts (those are text)
    """
    if dm.shape[1] < 3:
        raise ValueError("Expected Datamap sheet to have at least 3 columns.")

    c0, c1, c2 = dm.columns[:3]

    qtext: dict[str, str] = {}
    single_maps: dict[str, dict[str, str]] = {}
    multi_opts: dict[str, dict[str, str]] = {}
    order: list[str] = []

    current_var = None

    header_re = re.compile(r"^\[(?P<var>[^\]]+)\]\s*:\s*(?P<text>.*)$")
    opt_re = re.compile(r"^\[(?P<col>[^\]]+)\]$")

    # base var is everything before r<digits>...
    base_from_opt_re = re.compile(r"^(?P<base>.+?)r\d+.*$", re.IGNORECASE)

    for _, row in dm.iterrows():
        v0 = _s(row.get(c0))
        v1 = row.get(c1)
        v2 = _s(row.get(c2))

        # Header row
        m = header_re.match(v0)
        if m:
            current_var = m.group("var").strip()
            qtext[current_var] = m.group("text").strip()
            if current_var not in order:
                order.append(current_var)
            continue

        if not current_var:
            continue

        # Multi-select option row: col1 contains "[Q_...r#]"
        s1 = _s(v1)
        mopt = opt_re.match(s1)
        if mopt and v2:
            opt_col = mopt.group("col").strip()

            # Exclude open-ended text columns (…oe) from multi-select option set
            if opt_col.lower().endswith("oe"):
                continue

            mb = base_from_opt_re.match(opt_col)
            if mb:
                base = mb.group("base").strip()
            else:
                # fallback: if no r# pattern, associate to current var
                base = current_var

            multi_opts.setdefault(base, {})
            multi_opts[base][opt_col] = v2
            continue

        # Single-select mapping row: numeric-ish code -> label
        if v2 and pd.notna(v1):
            try:
                if isinstance(v1, float) and float(v1).is_integer():
                    code = str(int(v1))
                else:
                    code = str(v1).strip()

                if re.fullmatch(r"-?\d+(\.\d+)?", code):
                    if re.fullmatch(r"-?\d+\.0", code):
                        code = code[:-2]
                    single_maps.setdefault(current_var, {})
                    single_maps[current_var][code] = v2
            except Exception:
                pass

    return qtext, single_maps, multi_opts, order



# -----------------------------
# Decoding helper (fixes 1.0 vs "1" mismatches)
# -----------------------------
def decode_series(series: pd.Series, code_map: dict[str, str]) -> pd.Series:
    """
    Decode codes -> labels using string keys.
    Normalizes numeric codes so 1.0 matches "1".
    Keeps NaNs as NaN.
    """
    s = series.copy()
    is_na = s.isna()

    num = pd.to_numeric(s, errors="coerce")
    is_num = num.notna()

    keys = s.astype(str)
    int_like = is_num & (num % 1 == 0)
    keys = keys.where(~int_like, num.astype("Int64").astype(str))

    decoded = keys.map(code_map)
    decoded = decoded.where(decoded.notna(), keys)   # fallback to normalized key
    decoded = decoded.where(~is_na, np.nan)
    return decoded

def is_multiselect_option_col(var: str) -> bool:
    # matches things like Q_NEWS_LASTr12 or Q_NEWS_LASTr95oe
    return bool(re.search(r"r\d+", var, flags=re.IGNORECASE))

def base_from_r(var: str) -> str:
    # Q_NEWS_LASTr12 -> Q_NEWS_LAST
    return re.sub(r"r\d+.*$", "", var, flags=re.IGNORECASE)



# -----------------------------
# Weights
# -----------------------------
def get_weight_series(df: pd.DataFrame, use_weights: bool) -> pd.Series:
    if use_weights and "weight" in df.columns:
        return pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    return pd.Series(np.ones(len(df), dtype=float), index=df.index)


# -----------------------------
# Type inference helpers
# -----------------------------
def try_parse_datetime(series: pd.Series) -> tuple[pd.Series, float]:
    dt = pd.to_datetime(series, errors="coerce", infer_datetime_format=True, utc=False)
    return dt, float(dt.notna().mean())

def try_parse_numeric(series: pd.Series) -> tuple[pd.Series, float]:
    num = pd.to_numeric(series, errors="coerce")
    return num, float(num.notna().mean())

def normalize_selected_flag(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    ss = s.astype(str)
    return ss.isin(["1", "True", "true", "YES", "Yes", "Y", "y"]).fillna(False)


# -----------------------------
# Renderers
# -----------------------------
def render_age_birthyear_hist(df_filt: pd.DataFrame, w: pd.Series, use_weights: bool):
    # Interpret Q_AGE as birth year -> derive age
    current_year = datetime.now().year
    birth_year = pd.to_numeric(df_filt[AGE_VAR], errors="coerce").dropna()
    age = (current_year - birth_year).dropna()

    # Guardrails: drop nonsense ages
    age = age[(age >= 16) & (age <= 100)]
    w_age = w.loc[age.index]

    c1, c2 = st.columns(2)
    c1.metric("Unweighted n", f"{len(age):,}")
    c2.metric("Weighted n", f"{w_age.sum():,.2f}" if use_weights else "—")

    if age.empty:
        st.warning("No age values after filters.")
        return

    bin_size = st.slider("Bin size (years)", 1, 10, 5, key="age_bin_size")

    min_age = int(np.floor(age.min()))
    max_age = int(np.ceil(age.max()))
    start = min_age - (min_age % bin_size)
    bins = list(range(start, max_age + bin_size, bin_size))

    binned = pd.cut(age, bins=bins, right=False, include_lowest=True)
    hist = pd.DataFrame({"bin": binned, "w": w_age})
    hist = hist.groupby("bin", dropna=False)["w"].sum().reset_index()
    hist["bin_label"] = hist["bin"].astype(str)

    fig = px.bar(
        hist,
        x="bin_label",
        y="w",
        labels={"bin_label": "Age bin", "w": "Weighted count" if use_weights else "Count"},
        title="Age distribution (derived from birth year)",
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show underlying binned table", expanded=False):
        st.dataframe(hist, use_container_width=True)


def render_single_select(df_filt: pd.DataFrame, var: str, qlabel: str, single_maps: dict, w: pd.Series, use_weights: bool, key_prefix: str):
    if var not in df_filt.columns:
        st.warning("Column not found in CSV for this question.")
        return

    decoded = decode_series(df_filt[var], single_maps[var]) if var in single_maps else df_filt[var].astype(str).where(~df_filt[var].isna(), np.nan)

    tmp = pd.DataFrame({"response": decoded, "w": w}).dropna(subset=["response"])
    if tmp.empty:
        st.warning("No responses after filters.")
        return

    agg = tmp.groupby("response")["w"].sum().reset_index(name="weighted_n")
    agg["pct"] = agg["weighted_n"] / agg["weighted_n"].sum() if agg["weighted_n"].sum() else 0
    agg = agg.sort_values("pct", ascending=True)

    c1, c2 = st.columns(2)
    c1.metric("Unweighted n", f"{tmp.shape[0]:,}")
    c2.metric("Weighted n", f"{tmp['w'].sum():,.2f}" if use_weights else "—")

    fig = px.bar(
        agg,
        x="pct",
        y="response",
        orientation="h",
        text=agg["pct"].map(lambda x: f"{x:.0%}"),
        labels={"pct": "Share", "response": ""},
        title=qlabel,
    )
    fig.update_layout(xaxis_tickformat=".0%", height=max(320, 28 * len(agg)))
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_single_chart")

    with st.expander("Show table", expanded=False):
        st.dataframe(agg.sort_values("pct", ascending=False), use_container_width=True)


def render_multi_select(df_filt: pd.DataFrame, qlabel: str, opts: dict[str, str], w: pd.Series, use_weights: bool, key_prefix: str):
    cols = [c for c in opts.keys() if c in df_filt.columns]
    if not cols:
        st.warning("Multi-select question, but none of its option columns were found in the CSV.")
        return

    total_w = w.sum()
    rows = []
    for c in cols:
        sel = normalize_selected_flag(df_filt[c])
        sel_w = w[sel].sum()
        pct = (sel_w / total_w) if total_w else 0
        rows.append({"option": opts.get(c, c), "pct_selected": pct})

    out = pd.DataFrame(rows).sort_values("pct_selected", ascending=False)

    # Show top N to keep it readable; allow expanding to all
    top_n = st.slider("Show top N options", 5, min(40, len(out)), min(20, len(out)), key=f"{key_prefix}_topn")
    show_all = st.checkbox("Show all options", value=False, key=f"{key_prefix}_showall")
    plot_df = out if show_all else out.head(top_n)

    plot_df = plot_df.sort_values("pct_selected", ascending=True)

    c1, c2 = st.columns(2)
    c1.metric("Option columns", f"{len(cols):,}")
    c2.metric("Weighted n", f"{w.sum():,.2f}" if use_weights else "—")

    fig = px.bar(
        plot_df,
        x="pct_selected",
        y="option",
        orientation="h",
        text=plot_df["pct_selected"].map(lambda x: f"{x:.0%}"),
        labels={"pct_selected": "% selected", "option": ""},
        title=qlabel,
    )
    fig.update_layout(xaxis_tickformat=".0%", height=max(320, 24 * len(plot_df)))
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.caption("Multi-select: percent who selected each option (multiple responses allowed).")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show table (all options)", expanded=False):
        st.dataframe(out, use_container_width=True)


def render_fallback(df_filt: pd.DataFrame, var: str, qlabel: str, w: pd.Series, use_weights: bool, key_prefix: str):
    """
    For questions without a single-select map and not flagged as multi-select in the datamap:
      - datetime -> counts by date
      - numeric -> histogram
      - categorical (low-ish cardinality) -> bars
      - else -> searchable table
    """
    if var not in df_filt.columns:
        st.warning("Not a CSV column and not detected as multi-select.")
        return

    s = df_filt[var]

    # Try datetime
    if s.dtype == "object":
        dt, dt_ratio = try_parse_datetime(s)
        if dt_ratio >= 0.90:
            d = dt.dropna()
            daily = d.dt.date.value_counts().sort_index().rename_axis("date").reset_index(name="n")
            fig = px.bar(daily, x="date", y="n", labels={"n": "Responses", "date": "Date"}, title=qlabel)
            st.plotly_chart(fig, use_container_width=True)
            return

    # Try numeric
    num, num_ratio = try_parse_numeric(s)
    if num_ratio >= 0.90 and num.nunique(dropna=True) > 8:
        nn = num.dropna()
        c1, c2, c3 = st.columns(3)
        c1.metric("n", f"{len(nn):,}")
        c2.metric("Mean", f"{nn.mean():.2f}")
        c3.metric("Median", f"{nn.median():.2f}")
        fig = px.histogram(nn, nbins=30, title=qlabel)
        st.plotly_chart(fig, use_container_width=True)
        return

    # Categorical vs text
    nunique = s.nunique(dropna=True)
    if nunique <= 30:
        cat = s.astype(str).where(~s.isna(), np.nan)
        tmp = pd.DataFrame({"response": cat, "w": w}).dropna(subset=["response"])
        agg = tmp.groupby("response")["w"].sum().reset_index(name="weighted_n")
        agg["pct"] = agg["weighted_n"] / agg["weighted_n"].sum() if agg["weighted_n"].sum() else 0
        agg = agg.sort_values("pct", ascending=True)

        fig = px.bar(
            agg,
            x="pct",
            y="response",
            orientation="h",
            text=agg["pct"].map(lambda x: f"{x:.0%}"),
            labels={"pct": "Share", "response": ""},
            title=qlabel,
        )
        fig.update_layout(xaxis_tickformat=".0%", height=max(320, 28 * len(agg)))
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)
        return

    # Text/high-cardinality
    q = st.text_input("Search responses", key=f"{key_prefix}_search")
    if q:
        mask = s.astype(str).str.contains(q, case=False, na=False)
        st.write(f"Matches: {mask.sum():,}")
        st.dataframe(pd.DataFrame({"response": s[mask]}).head(500), use_container_width=True)
    else:
        st.dataframe(pd.DataFrame({"response": s}).head(200), use_container_width=True)


# -----------------------------
# App boot
# -----------------------------
df = load_poll(CSV_PATH)
dm = load_datamap(DATAMAP_PATH)
qtext, single_maps, multi_opts, order = parse_datamap(dm)

# Sidebar: filters and weights
with st.sidebar:
    st.header("Filters")
    use_weights = st.toggle("Use weights (if available)", value=("weight" in df.columns))
    selections = {}

    for label, var in FILTER_VARS.items():
        if var not in df.columns:
            st.caption(f"⚠️ Missing column in CSV: {var}")
            selections[var] = []
            continue

        decoded = decode_series(df[var], single_maps[var]) if var in single_maps else df[var].astype(str).where(~df[var].isna(), np.nan)
        options = sorted([x for x in decoded.dropna().unique().tolist() if str(x).strip() != ""], key=lambda x: str(x))
        selections[var] = st.multiselect(label, options=options, default=[])

# Apply filters
df_filt = df.copy()
for var, selected in selections.items():
    if not selected or var not in df_filt.columns:
        continue
    decoded = decode_series(df_filt[var], single_maps[var]) if var in single_maps else df_filt[var].astype(str).where(~df_filt[var].isna(), np.nan)
    df_filt = df_filt.loc[decoded.isin(selected)]

w = get_weight_series(df_filt, use_weights)
use_weights_flag = bool(use_weights and ("weight" in df_filt.columns))

# Summary metrics
c1, c2 = st.columns(2)
c1.metric("Unweighted n (after filters)", f"{len(df_filt):,}")
c2.metric("Weighted n (after filters)", f"{w.sum():,.2f}" if use_weights_flag else "—")

st.divider()

# -----------------------------
# Render questions (all, in order)
# -----------------------------
# Put Q_AGE first, then rest (if it exists)
render_list = []
if AGE_VAR in order or AGE_VAR in df.columns:
    render_list.append(AGE_VAR)
for v in order:
    if v != AGE_VAR:
        render_list.append(v)

for var in render_list:
    qlabel = qtext.get(var, var)
    title = f"{var} — {qlabel}" if qlabel and qlabel != var else var
    key_prefix = f"q_{var}"

    with st.expander(title, expanded=(var in [AGE_VAR])):
        # Special case: Q_AGE
        if var == AGE_VAR:
            if AGE_VAR not in df_filt.columns:
                st.warning("Q_AGE is not present as a CSV column.")
            else:
                render_age_birthyear_hist(df_filt, w, use_weights_flag)

        # Multi-select (datamap says so)
        elif var in multi_opts:
            render_multi_select(df_filt, qlabel, multi_opts[var], w, use_weights_flag, key_prefix=key_prefix)

        # Single-select (mapped)
        elif var in single_maps:
            render_single_select(df_filt, var, qlabel, single_maps, w, use_weights_flag)

        # Fallback inference
        else:
            render_fallback(df_filt, var, qlabel, w, use_weights_flag, key_prefix=key_prefix)

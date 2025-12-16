# uw_housing_model.py
"""
UW HFS Housing Structural Risk Model (Index-Based, Outcome Neutral)
v4.1 - Fix Altair "collapsed charts" bug + add UW Enrollment Trend lever

Fix:
- Charts collapsing under scenarios with a debt-peak band (e.g., Structural Squeeze) was caused by the
  peak-band layer not sharing the same explicit x-scale domain as the rest of the chart.
- In layered Vega-Lite specs, an unbounded rect layer can end up influencing the shared x-scale in
  unexpected ways, causing the rest of the data to render in a narrow slice.
- v4.1 enforces a shared x-scale domain on the band layer and uses alt.layer(...).resolve_scale(x="shared").

New feature:
- Added "UW Enrollment Trend (Class Size)" slider (range -20% to +10%, default 0%).
- Demand engine update:
    Demand = Demographics * Enrollment_Trend * Housing_Preference
  Enrollment trend is implemented as a constant multiplier across the horizon (simple and explicit).
  If you later want it to ramp "by 2035", we can match the other sliders.

What did NOT change:
- No absolute dollars are computed, displayed, or exported.
- DSCR approximation logic and index mechanics are retained.
- Outcome neutrality: Baseline remains non-alarmist, flat debt timing.

Run:
    streamlit run uw_housing_model.py
"""

from __future__ import annotations

from io import StringIO
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Streamlit stability requirement: must be the first Streamlit command.
st.set_page_config(
    page_title="UW HFS Housing Structural Risk Dashboard (Index-Based)",
    layout="wide",
)

# Altair is typically present in Streamlit environments, but keep a safe fallback.
try:
    import altair as alt

    HAS_ALTAIR = True
except Exception:
    HAS_ALTAIR = False


# -----------------------------
# Hardcoded reconciled inputs
# -----------------------------
BASE_YEAR = 2025
END_YEAR = 2045
YEARS = list(range(BASE_YEAR, END_YEAR + 1))

RECONCILED_DATA = {
    "_metadata": {
        "reconciliation_date": "2025-12-15",
        "note": "Financial values here are for BASELINE INDEXING only.",
    },
    "housing_portfolio": {
        "totals": {
            "total_operating_capacity": {"value": 9114},
            "overflow_beds": {"value": 879},
        },
        "occupancy": {
            "current_rate": {"value": 1.004, "year": "2025-26"},
            "current_headcount": {"value": 9149},
        },
        "planned_changes": {
            "haggett_hall_replacement": {
                "status": "under_construction",
                "projected_opening": "Fall 2027",
                "planned_beds": 800,
            }
        },
    },
    "demographics": {
        "source_note": "WA OFM 18-year-old Population Projections",
        "wa_18yo_population": [
            {"year": 2025, "population": 101845},
            {"year": 2026, "population": 103205},
            {"year": 2027, "population": 104050},
            {"year": 2028, "population": 104473},
            {"year": 2029, "population": 104553},
            {"year": 2030, "population": 104571},
            {"year": 2031, "population": 104324},
            {"year": 2032, "population": 103789},
            {"year": 2033, "population": 102995},
            {"year": 2034, "population": 101993},
            {"year": 2035, "population": 101521},
            {"year": 2036, "population": 101513},
            {"year": 2037, "population": 101908},
            {"year": 2038, "population": 102390},
            {"year": 2039, "population": 97490},
            {"year": 2040, "population": 99750},
            {"year": 2041, "population": 97327},
            {"year": 2042, "population": 96030},
            {"year": 2043, "population": 97471},
            {"year": 2044, "population": 98124},
            {"year": 2045, "population": 98770},
        ],
    },
    "financial_ratios": {
        "base_dscr": {"value": 1.57, "note": "2022 Actual, serves as DSCR anchor"},
        "required_dscr": {"value": 1.25},
        "debt_service_share": {
            "value": 0.35,
            "note": "Est. debt service as % of base-year indexed revenue (diagnostic only)",
        },
        "expense_share": {
            "value": 0.50,
            "note": "Est. operating expense as % of base-year indexed revenue",
        },
        "margin_share": {"value": 0.15, "note": "Net margin share (context only)"},
    },
}

WA_OFM_18YO = [(x["year"], x["population"]) for x in RECONCILED_DATA["demographics"]["wa_18yo_population"]]

DEBT_SHAPES = [
    "Flat (Baseline)",
    "Front-Loaded",
    'The "Cliff" (Risk)',
    "Custom",
]

# Scenarios remain outcome-neutral by default.
SCENARIOS: Dict[str, Dict[str, object]] = {
    "Baseline": {
        "debt_shape": "Flat (Baseline)",
        "rate_escalation_pct": 2.5,
        "expense_inflation_pct": 2.5,
        "national_trend_pct_by_2035": 0,
        "wa_demand_share": 70,  # percent
        "uw_enrollment_trend_pct": 0,  # NEW lever (percent)
        "behavior_headwind_pct_by_2035": 0,
        "haggett_net_beds": 0,
        "expense_share_pct": int(RECONCILED_DATA["financial_ratios"]["expense_share"]["value"] * 100),
        "debt_share_pct": int(RECONCILED_DATA["financial_ratios"]["debt_service_share"]["value"] * 100),
        "custom_peak_multiplier": 1.20,
        "custom_peak_year": 2030,
        "focus_year": 2035,
    },
    "Structural Squeeze": {
        "debt_shape": 'The "Cliff" (Risk)',
        "rate_escalation_pct": 3.0,
        "expense_inflation_pct": 4.0,
        "national_trend_pct_by_2035": 0,
        "wa_demand_share": 70,
        "uw_enrollment_trend_pct": 0,  # NEW lever
        "behavior_headwind_pct_by_2035": 0,
        "haggett_net_beds": 0,
        "expense_share_pct": int(RECONCILED_DATA["financial_ratios"]["expense_share"]["value"] * 100),
        "debt_share_pct": int(RECONCILED_DATA["financial_ratios"]["debt_service_share"]["value"] * 100),
        "custom_peak_multiplier": 1.20,
        "custom_peak_year": 2030,
        "focus_year": 2035,
    },
    "Demographic Trough": {
        "debt_shape": "Flat (Baseline)",
        "rate_escalation_pct": 2.5,
        "expense_inflation_pct": 2.5,
        "national_trend_pct_by_2035": -10,
        "wa_demand_share": 70,
        "uw_enrollment_trend_pct": 0,  # NEW lever
        "behavior_headwind_pct_by_2035": 0,
        "haggett_net_beds": 0,
        "expense_share_pct": int(RECONCILED_DATA["financial_ratios"]["expense_share"]["value"] * 100),
        "debt_share_pct": int(RECONCILED_DATA["financial_ratios"]["debt_service_share"]["value"] * 100),
        "custom_peak_multiplier": 1.20,
        "custom_peak_year": 2030,
        "focus_year": 2035,
    },
    "Bond Stress Test (Illustrative)": {
        "debt_shape": 'The "Cliff" (Risk)',
        "rate_escalation_pct": 2.5,
        "expense_inflation_pct": 4.5,
        "national_trend_pct_by_2035": -10,
        "wa_demand_share": 70,
        "uw_enrollment_trend_pct": -10,  # NEW lever used in stress test
        "behavior_headwind_pct_by_2035": -15,
        "haggett_net_beds": 0,
        "expense_share_pct": int(RECONCILED_DATA["financial_ratios"]["expense_share"]["value"] * 100),
        "debt_share_pct": int(RECONCILED_DATA["financial_ratios"]["debt_service_share"]["value"] * 100),
        "custom_peak_multiplier": 1.20,
        "custom_peak_year": 2030,
        "focus_year": 2035,
    },
    "Custom (Keep Current Settings)": {},
}

SCENARIO_CONTEXT: Dict[str, str] = {
    "Baseline": "Baseline: steady rent and cost growth, flat debt timing, no macro or preference headwinds.",
    "Structural Squeeze": "Structural Squeeze: faster cost growth plus a 2030–2037 debt peak (the cliff).",
    "Demographic Trough": "Demographic Trough: WA pipeline follows OFM, non-resident pipeline shrinks 10% by 2035.",
    "Bond Stress Test (Illustrative)": "Illustrative stress: cost growth + debt peak + weaker demand, designed to show what a breach looks like.",
    "Custom (Keep Current Settings)": "Custom: keeps your current slider settings (no automatic resets).",
}


# -----------------------------
# Numerical safety helpers
# -----------------------------
def safe_div(n, d, default=np.nan):
    """Division that never throws divide-by-zero, and never returns inf/-inf."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(n, d)

    if np.isscalar(out):
        return out if np.isfinite(out) else default

    out = np.where(np.isfinite(out), out, default)
    return out


def clamp(x, lo=None, hi=None):
    lo = -np.inf if lo is None else lo
    hi = np.inf if hi is None else hi
    return np.clip(x, lo, hi)


def linear_progress(years, start_year, end_year):
    """0 at start_year, 1 at end_year, clamped outside the window."""
    years = np.asarray(years, dtype=float)
    denom = max(1.0, float(end_year - start_year))
    prog = (years - float(start_year)) / denom
    return clamp(prog, 0.0, 1.0)


def finite_minmax(vals: np.ndarray) -> Tuple[float, float]:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan)
    return float(np.min(v)), float(np.max(v))


def padded_domain(
    vals: np.ndarray,
    pad_abs: float,
    force_include: Optional[Tuple[float, float]] = None,
    clamp_low: Optional[float] = None,
    clamp_high: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """Y-axis domain helper that uses chart space efficiently while including key thresholds."""
    vmin, vmax = finite_minmax(vals)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None

    lo = vmin - pad_abs
    hi = vmax + pad_abs

    if force_include is not None:
        lo = min(lo, float(force_include[0]))
        hi = max(hi, float(force_include[1]))

    if clamp_low is not None:
        lo = max(lo, float(clamp_low))
    if clamp_high is not None:
        hi = min(hi, float(clamp_high))

    if hi <= lo:
        hi = lo + max(1e-6, pad_abs)

    return (lo, hi)


# -----------------------------
# Demand and debt profile builders
# -----------------------------
def build_ofm_df() -> pd.DataFrame:
    df = pd.DataFrame(WA_OFM_18YO, columns=["year", "wa_18yo_population"]).copy()
    df = df[(df["year"] >= BASE_YEAR) & (df["year"] <= END_YEAR)].sort_values("year")
    base_pop = float(df.loc[df["year"] == BASE_YEAR, "wa_18yo_population"].iloc[0])
    df["wa_18yo_index"] = safe_div(df["wa_18yo_population"], base_pop, default=0.0) * 100.0
    return df.reset_index(drop=True)


def build_national_index(years, pct_by_2035: float) -> np.ndarray:
    """Non-resident macro proxy ramps to 2035 then holds constant."""
    prog = linear_progress(years, BASE_YEAR, 2035)
    idx = 100.0 * (1.0 + pct_by_2035 * prog)
    return clamp(idx, 0.0, None)


def build_behavior_index(years, pct_by_2035: float) -> np.ndarray:
    """Housing preference shift ramps to 2035 then holds constant."""
    prog = linear_progress(years, BASE_YEAR, 2035)
    idx = 100.0 * (1.0 + pct_by_2035 * prog)
    return clamp(idx, 0.0, None)


def build_enrollment_index_constant(years, pct_level: float) -> np.ndarray:
    """
    NEW: UW Enrollment Trend (Class Size).
    Implemented as a constant level multiplier across the horizon.

    Example: -10% means enrollment pipeline is 90 (index) every year.
    """
    idx = 100.0 * (1.0 + pct_level)
    return np.full(len(years), clamp(idx, 0.0, None), dtype=float)


def build_capacity_headcount(years, haggett_net_beds: int) -> np.ndarray:
    """Capacity is used ONLY as a physical cap on occupancy headcount."""
    base_operating = int(RECONCILED_DATA["housing_portfolio"]["totals"]["total_operating_capacity"]["value"])
    overflow = int(RECONCILED_DATA["housing_portfolio"]["totals"]["overflow_beds"]["value"])
    cap_base = float(base_operating + overflow)

    cap = np.full(len(years), cap_base, dtype=float)
    cap = np.where(np.asarray(years) >= 2027, cap + float(haggett_net_beds), cap)
    return clamp(cap, 0.0, None)


def build_debt_index(
    years: np.ndarray,
    shape: str,
    custom_peak_multiplier: float,
    custom_peak_year: int,
) -> np.ndarray:
    """Debt service timing expressed as an index (2025 = 100). No dollars."""
    years = np.asarray(years, dtype=int)
    debt = np.full(len(years), 100.0, dtype=float)

    if shape == "Flat (Baseline)":
        return debt

    if shape == "Front-Loaded":
        start, end = BASE_YEAR, 2035
        slope = (80.0 - 100.0) / max(1, (end - start))
        for i, y in enumerate(years):
            if y <= end:
                debt[i] = 100.0 + slope * (y - start)
            else:
                debt[i] = 80.0
        return clamp(debt, 1.0, None)

    if shape == 'The "Cliff" (Risk)':
        for i, y in enumerate(years):
            if 2030 <= y <= 2037:
                debt[i] = 120.0
            elif y >= 2038:
                debt[i] = 80.0
            else:
                debt[i] = 100.0
        return clamp(debt, 1.0, None)

    if shape == "Custom":
        peak_mult = float(custom_peak_multiplier)
        peak_mult = clamp(peak_mult, 0.5, 2.0)
        peak_start = int(custom_peak_year)
        peak_end = peak_start + 7
        for i, y in enumerate(years):
            if peak_start <= y <= peak_end:
                debt[i] = 100.0 * peak_mult
            elif y > peak_end:
                debt[i] = 80.0
            else:
                debt[i] = 100.0
        return clamp(debt, 1.0, None)

    return debt


def get_debt_peak_window(
    debt_shape: str,
    custom_peak_year: int,
    custom_peak_multiplier: float,
) -> Optional[Tuple[int, int]]:
    """UI helper ONLY: identify a 'peak pressure window' for shading charts."""
    if debt_shape == 'The "Cliff" (Risk)':
        return (2030, 2037)
    if debt_shape == "Custom" and float(custom_peak_multiplier) > 1.0001:
        start = int(custom_peak_year)
        return (start, start + 7)
    return None


# -----------------------------
# Core model (indices only)
# -----------------------------
def run_model(
    wa_demand_share: float,
    national_trend_pct_by_2035: float,
    uw_enrollment_trend_pct: float,
    behavior_headwind_pct_by_2035: float,
    haggett_net_beds: int,
    rate_escalation: float,
    expense_inflation: float,
    expense_share: float,
    debt_share: float,
    debt_shape: str,
    custom_peak_multiplier: float,
    custom_peak_year: int,
) -> pd.DataFrame:
    """
    Builds a year-by-year dataframe of indices and DSCR-derived metrics.

    Key constraint:
    - No absolute dollars. Indices only.

    DSCR approximation (directional, anchored):
    - DSCR(t) ≈ Base_DSCR × (NOI_Index(t) / Debt_Index(t))
    """
    df = build_ofm_df()
    years = df["year"].to_numpy(dtype=int)
    t = (years - BASE_YEAR).astype(int)

    national_index = build_national_index(years, national_trend_pct_by_2035)
    behavior_index = build_behavior_index(years, behavior_headwind_pct_by_2035)
    enrollment_index = build_enrollment_index_constant(years, uw_enrollment_trend_pct)

    # Blend WA and non-resident demand environments (weighting avoids false precision).
    wa_share = float(clamp(wa_demand_share, 0.0, 1.0))
    demographic_index = wa_share * df["wa_18yo_index"].to_numpy(dtype=float) + (1.0 - wa_share) * national_index

    # Demand index includes:
    # - demographics (WA + non-resident blend)
    # - UW enrollment trend (policy / admissions / budget driven)
    # - housing preference shift (competition / take-rate driven)
    #
    # All three are base-100 indices, so multiply and divide by 100^2 to stay base-100.
    demand_index = safe_div(demographic_index * enrollment_index * behavior_index, 10000.0, default=0.0)
    demand_index = clamp(demand_index, 0.0, None)

    # Convert demand index to headcount, then cap by physical capacity.
    base_headcount = float(RECONCILED_DATA["housing_portfolio"]["occupancy"]["current_headcount"]["value"])
    demand_headcount = base_headcount * safe_div(demand_index, 100.0, default=0.0)

    capacity_headcount = build_capacity_headcount(years, haggett_net_beds)
    occupied_headcount = np.minimum(demand_headcount, capacity_headcount)
    occupied_headcount = clamp(occupied_headcount, 0.0, None)

    occupancy_index = safe_div(occupied_headcount, base_headcount, default=0.0) * 100.0

    # Indices (still no dollars)
    revenue_index = occupancy_index * np.power(1.0 + float(rate_escalation), t)
    expense_index = 100.0 * np.power(1.0 + float(expense_inflation), t)

    debt_index = build_debt_index(
        years=years,
        shape=debt_shape,
        custom_peak_multiplier=custom_peak_multiplier,
        custom_peak_year=custom_peak_year,
    )

    exp_share = float(clamp(expense_share, 0.0, 1.0))
    net_operating_index = revenue_index - (exp_share * expense_index)

    # NOI_Index rebases net operating to 2025 = 100 for stable ratio math.
    net_operating_base = 100.0 * (1.0 - exp_share)
    noi_index = safe_div(net_operating_index, net_operating_base, default=np.nan) * 100.0

    base_dscr = float(RECONCILED_DATA["financial_ratios"]["base_dscr"]["value"])
    dscr_est = base_dscr * safe_div(noi_index, debt_index, default=np.nan)

    required_dscr = float(RECONCILED_DATA["financial_ratios"]["required_dscr"]["value"])
    base_cushion = base_dscr - required_dscr
    safety_margin_pct = safe_div((dscr_est - required_dscr), base_cushion, default=np.nan) * 100.0

    debt_sh = float(clamp(debt_share, 0.0, 1.0))
    relative_coverage_ratio = safe_div(net_operating_index, (debt_sh * debt_index), default=np.nan)

    out = pd.DataFrame(
        {
            "year": years,
            "WA_18yo_Population": df["wa_18yo_population"].astype(int),
            "WA_18yo_Index": df["wa_18yo_index"].astype(float),
            "National_Global_Index": national_index.astype(float),
            "Enrollment_Index": enrollment_index.astype(float),  # NEW
            "Behavior_Index": behavior_index.astype(float),
            "Demographic_Index": demographic_index.astype(float),
            "Demand_Index": demand_index.astype(float),
            "Capacity_Headcount_Cap": capacity_headcount.astype(float),
            "Capacity_Index": (safe_div(capacity_headcount, base_headcount, default=np.nan) * 100.0).astype(float),
            "Occupied_Headcount": occupied_headcount.astype(float),
            "Occupancy_Index": occupancy_index.astype(float),
            "Revenue_Index": revenue_index.astype(float),
            "Expense_Index": expense_index.astype(float),
            "Debt_Index": debt_index.astype(float),
            "Net_Operating_Index": net_operating_index.astype(float),
            "NOI_Index": noi_index.astype(float),
            "Relative_Coverage_Ratio": relative_coverage_ratio.astype(float),
            "DSCR_Est": dscr_est.astype(float),
            "Safety_Margin_%": safety_margin_pct.astype(float),
        }
    )

    out["Covenant_Breach"] = out["DSCR_Est"] < required_dscr
    out["Headroom_Above_Min_DSCR"] = out["DSCR_Est"] - required_dscr
    return out


def value_at_year(df: pd.DataFrame, year: int, col: str, default=np.nan) -> float:
    row = df[df["year"] == year]
    if row.empty:
        return default
    return float(row.iloc[0][col])


def first_breach_year(df: pd.DataFrame) -> Optional[int]:
    future = df[df["year"] >= BASE_YEAR].copy()
    breach = future[future["Covenant_Breach"]]
    if breach.empty:
        return None
    return int(breach.iloc[0]["year"])


def worst_year_by(df: pd.DataFrame, col: str) -> Optional[int]:
    s = df[["year", col]].copy()
    s = s[np.isfinite(s[col])]
    if s.empty:
        return None
    i = int(s[col].idxmin())
    return int(df.loc[i, "year"])


# -----------------------------
# UI helpers (formatting only)
# -----------------------------
def fmt_num(x: float, fmt: str, na: str = "n/a") -> str:
    return fmt.format(x) if np.isfinite(x) else na


def scenario_defaults_for(name: str) -> Dict[str, object]:
    return dict(SCENARIOS.get(name, {}))


def apply_scenario_defaults(name: str) -> None:
    defaults = scenario_defaults_for(name)
    for k, v in defaults.items():
        st.session_state[k] = v


def on_scenario_change() -> None:
    apply_scenario_defaults(st.session_state.get("scenario", "Baseline"))


def traffic_status_line(ok: bool, ok_text: str, bad_text: str) -> str:
    return f"🟢 {ok_text}" if ok else f"🔴 {bad_text}"


# Altair x-axis helpers
def year_axis():
    # Fixed ticks prevent “nice” ticks from bleeding into 2024/2046.
    return alt.Axis(values=list(range(BASE_YEAR, END_YEAR + 1, 2)), format="d")


def x_year(title="Year"):
    # Explicit scale domain is critical for stable rendering across layered charts.
    return alt.X(
        "year:Q",
        title=title,
        axis=year_axis(),
        scale=alt.Scale(domain=[BASE_YEAR, END_YEAR], nice=False),
    )


def peak_band_layer(peak_window: Optional[Tuple[int, int]]):
    """
    BUGFIX:
    The debt-peak shading band MUST share the same x-scale domain as the other layers.
    If not, Vega-Lite can infer an incorrect shared x-scale domain and squash other layers.

    Implementation:
    - Use a tiny dataframe with year_start/year_end (quantitative)
    - Apply the same explicit x scale domain
    - Axis is None so only the “real” chart layer owns the axis rendering
    """
    if peak_window is None:
        return None

    band_df = pd.DataFrame({"year_start": [peak_window[0]], "year_end": [peak_window[1]]})
    return (
        alt.Chart(band_df)
        .mark_rect(opacity=0.08, color="#f59e0b")
        .encode(
            x=alt.X(
                "year_start:Q",
                scale=alt.Scale(domain=[BASE_YEAR, END_YEAR], nice=False),
                axis=None,
            ),
            x2=alt.X2("year_end:Q"),
        )
    )


# -----------------------------
# Page header
# -----------------------------
st.title("UW HFS Housing Structural Risk Dashboard")
st.caption("Index-based, outcome-neutral decision support. No currency values are computed or exported.")

base_dscr = float(RECONCILED_DATA["financial_ratios"]["base_dscr"]["value"])
required_dscr = float(RECONCILED_DATA["financial_ratios"]["required_dscr"]["value"])

with st.expander("About This Model (Plain English)", expanded=False):
    st.markdown(
        f"""
**What is a bond covenant?**  
A covenant is a rule in the bond agreement. For HFS, one key covenant is a minimum coverage requirement:
we must keep the **coverage ratio** above **{required_dscr:.2f}**.

**What is “coverage” (DSCR)?**  
DSCR stands for **Debt Service Coverage Ratio**. It is a standard lender ratio:
**cash available for debt payments ÷ required debt payments**.

**Why do we need a safety buffer?**  
Because DSCR moves with enrollment, pricing, costs, and debt timing. If DSCR drops below the minimum, it’s a covenant breach.

**Important note on the math (directional lens):**  
Coverage is estimated using relative indices anchored to today’s DSCR:
DSCR(t) ≈ Base DSCR × NOI_Index(t) ÷ Debt_Index(t).
        """.strip()
    )

with st.expander("How to Use This Dashboard (Fast)", expanded=False):
    st.markdown(
        """
- Start with Baseline (it should not look like a crisis).
- If you want to see what “bad” looks like quickly, select “Bond Stress Test (Illustrative)”.
- The most important visual is “Headroom Above Minimum”:
  - above 0 = meeting the bond requirement
  - at 0 = right on the line
  - below 0 = below the requirement (breach)
- Use the Focus Year slider for the KPI snapshot, or click “Jump to worst year”.
        """.strip()
    )

# -----------------------------
# State initialization + migration
# -----------------------------
if "initialized" not in st.session_state:
    st.session_state["scenario"] = "Baseline"
    for k, v in SCENARIOS["Baseline"].items():
        st.session_state[k] = v
    st.session_state["initialized"] = True

# Migration safeguard from older versions (where WA share may have been stored as 0.70)
if "wa_demand_share" in st.session_state:
    v = st.session_state["wa_demand_share"]
    if isinstance(v, float) and 0.0 <= v <= 1.0:
        st.session_state["wa_demand_share"] = int(round(v * 100))

# Ensure new key exists for users coming from older session state
if "uw_enrollment_trend_pct" not in st.session_state:
    st.session_state["uw_enrollment_trend_pct"] = 0

if "focus_year" not in st.session_state:
    st.session_state["focus_year"] = 2035


# -----------------------------
# Sidebar (controls)
# -----------------------------
with st.sidebar:
    st.header("Controls")

    st.selectbox(
        "Scenario (One-click)",
        options=list(SCENARIOS.keys()),
        key="scenario",
        on_change=on_scenario_change,
        help="Loads the assumptions for the selected scenario. Use reset to undo experiments without refreshing the whole page.",
    )

    scenario_name = str(st.session_state.get("scenario", "Baseline"))
    st.info(SCENARIO_CONTEXT.get(scenario_name, "Scenario loaded."))

    defaults_exist = len(scenario_defaults_for(scenario_name)) > 0
    if st.button(
        "Reset to Scenario Defaults",
        disabled=not defaults_exist,
        help="Reverts sliders to the defaults for the currently selected scenario (does not change the scenario).",
        use_container_width=True,
    ):
        apply_scenario_defaults(scenario_name)
        st.success("Scenario defaults reloaded.")

    st.subheader("KPI Snapshot Year")

    st.slider(
        "Focus Year",
        min_value=BASE_YEAR,
        max_value=END_YEAR,
        step=1,
        key="focus_year",
        help=(
            "Controls the year used for the KPI cards. Default is 2035 as a common planning horizon. "
            "You can change it anytime."
        ),
    )

    st.subheader("Enrollment & Preference")

    st.slider(
        "Enrollment Source Weighting",
        min_value=20,
        max_value=90,
        step=1,
        key="wa_demand_share",
        format="%d%%",
        help=(
            "Units: percent.\n\n"
            "Blends WA demographics with non-resident pipeline conditions.\n"
            "Example: 70% means 70% of demand follows WA OFM, 30% follows non-resident conditions."
        ),
    )

    # NEW SLIDER
    st.slider(
        "UW Enrollment Trend (Class Size)",
        min_value=-20,
        max_value=10,
        step=1,
        key="uw_enrollment_trend_pct",
        format="%d%%",
        help=(
            "Changes in total freshman class size due to University policy, budget cuts, or acceptance rates. "
            "Independent of housing preference.\n\n"
            "Units: percent.\n"
            "Example: -10% means the overall student pipeline is 10% smaller than baseline (all years)."
        ),
    )

    st.slider(
        "Non-Resident Pipeline Trend (by 2035)",
        min_value=-50,
        max_value=50,
        step=1,
        key="national_trend_pct_by_2035",
        format="%d%%",
        help=(
            "Units: percent change by 2035 relative to 2025.\n\n"
            "Example: -10% means non-resident pipeline index moves from 100 (2025) to 90 (2035), "
            "then holds at 90 after 2035."
        ),
    )

    st.slider(
        "Housing Preference Shift (by 2035)",
        min_value=-50,
        max_value=20,
        step=1,
        key="behavior_headwind_pct_by_2035",
        format="%d%%",
        help=(
            "Units: percent change by 2035 relative to 2025 applied on top of demographics.\n\n"
            "Example: -15% means that by 2035, demand is 85% of what demographics alone would imply "
            "(more students choosing off-campus options)."
        ),
    )

    st.slider(
        "New Capacity (Net Beds) (effective 2027)",
        min_value=-500,
        max_value=1500,
        step=25,
        key="haggett_net_beds",
        help=(
            "Units: beds (headcount cap).\n\n"
            "Does not create demand. Only changes the maximum students we can house."
        ),
    )

    st.subheader("Prices & Costs")

    st.slider(
        "Annual Rent Increase",
        min_value=0.0,
        max_value=8.0,
        step=0.1,
        key="rate_escalation_pct",
        format="%.1f%%",
        help="Units: percent per year. Applies as compounding growth on top of occupancy changes.",
    )

    st.slider(
        "Annual Cost Growth",
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        key="expense_inflation_pct",
        format="%.1f%%",
        help="Units: percent per year. Applies as compounding annual growth (salaries, utilities, COGS).",
    )

    st.subheader("Debt")

    st.selectbox(
        "Debt Service Timing",
        options=DEBT_SHAPES,
        key="debt_shape",
        help="Models the timing pattern of debt payments as an index (2025 = 100), not a dollar schedule.",
    )

    if st.session_state["debt_shape"] == "Custom":
        st.slider(
            "Peak Debt Multiplier",
            min_value=1.00,
            max_value=1.80,
            step=0.01,
            key="custom_peak_multiplier",
            help="Multiplier applied during the 8-year peak window (e.g., 1.20 = +20%).",
        )
        st.slider(
            "Peak Year",
            min_value=2027,
            max_value=2040,
            step=1,
            key="custom_peak_year",
            help="Start year of the 8-year peak window.",
        )
    elif st.session_state["debt_shape"] == 'The "Cliff" (Risk)':
        st.caption("Cliff definition: Debt Index = 120 for 2030–2037, then 80 from 2038 onward.")

    with st.expander("Advanced: Base Shares (Index Anchors)", expanded=False):
        st.slider(
            "Operating Expense Share (base year)",
            min_value=30,
            max_value=70,
            step=1,
            key="expense_share_pct",
            format="%d%%",
        )
        st.slider(
            "Debt Service Share (base year) (diagnostic)",
            min_value=10,
            max_value=50,
            step=1,
            key="debt_share_pct",
            format="%d%%",
        )

    with st.expander("Reference: Bond Minimum Requirement (read-only)", expanded=False):
        st.write(f"Base DSCR (anchor): {base_dscr:.2f}")
        st.write(f"Bond minimum requirement (covenant): {required_dscr:.2f}")


# -----------------------------
# Collect params (convert UI % to decimals)
# -----------------------------
params = {
    "wa_demand_share": float(st.session_state["wa_demand_share"]) / 100.0,
    "national_trend_pct_by_2035": float(st.session_state["national_trend_pct_by_2035"]) / 100.0,
    "uw_enrollment_trend_pct": float(st.session_state["uw_enrollment_trend_pct"]) / 100.0,  # NEW
    "behavior_headwind_pct_by_2035": float(st.session_state["behavior_headwind_pct_by_2035"]) / 100.0,
    "haggett_net_beds": int(st.session_state["haggett_net_beds"]),
    "rate_escalation": float(st.session_state["rate_escalation_pct"]) / 100.0,
    "expense_inflation": float(st.session_state["expense_inflation_pct"]) / 100.0,
    "expense_share": float(st.session_state["expense_share_pct"]) / 100.0,
    "debt_share": float(st.session_state["debt_share_pct"]) / 100.0,
    "debt_shape": str(st.session_state["debt_shape"]),
    "custom_peak_multiplier": float(st.session_state.get("custom_peak_multiplier", 1.20)),
    "custom_peak_year": int(st.session_state.get("custom_peak_year", 2030)),
}

df = run_model(**params)

focus_year = int(st.session_state.get("focus_year", 2035))
focus_year = int(clamp(focus_year, BASE_YEAR, END_YEAR))

breach_year = first_breach_year(df)
worst_year = worst_year_by(df, "Headroom_Above_Min_DSCR")

peak_window = get_debt_peak_window(
    debt_shape=params["debt_shape"],
    custom_peak_year=params["custom_peak_year"],
    custom_peak_multiplier=params["custom_peak_multiplier"],
)

# Focus-year values
dscr_focus = value_at_year(df, focus_year, "DSCR_Est", default=np.nan)
headroom_focus = value_at_year(df, focus_year, "Headroom_Above_Min_DSCR", default=np.nan)

# Worst-year values
headroom_min = float(np.nanmin(df["Headroom_Above_Min_DSCR"].to_numpy(dtype=float)))

dscr_ok_focus = bool(np.isfinite(dscr_focus) and dscr_focus >= required_dscr)
headroom_ok_focus = bool(np.isfinite(headroom_focus) and headroom_focus >= 0.0)

# Compliance banner
if breach_year is None:
    st.success(f"Bond minimum requirement is met in all years shown ({BASE_YEAR}–{END_YEAR}) under the current assumptions.")
else:
    st.error(
        f"Bond minimum requirement is breached starting in {breach_year} under the current assumptions. "
        "Use the charts below to see whether the driver is demand, costs, or debt timing."
    )

# KPI row
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Bond requirement status", f"No breach through {END_YEAR}" if breach_year is None else f"Breach starts {breach_year}")

with k2:
    st.metric(f"Bond coverage ratio in {focus_year}", fmt_num(dscr_focus, "{:.2f}"))
    st.caption(traffic_status_line(dscr_ok_focus, f"Above minimum ({required_dscr:.2f})", f"Below minimum ({required_dscr:.2f})"))

with k3:
    st.metric(f"Headroom above minimum in {focus_year}", fmt_num(headroom_focus, "{:+.2f}"))
    st.caption(traffic_status_line(headroom_ok_focus, "Headroom is positive", "Headroom is negative (breach)"))

with k4:
    st.metric("Worst-year headroom (all years)", fmt_num(headroom_min, "{:+.2f}"))
    if worst_year is not None:
        st.caption(f"Worst year: {worst_year}")

if worst_year is not None:
    if st.button("Jump to worst year (sets Focus Year)", use_container_width=False):
        st.session_state["focus_year"] = worst_year
        st.rerun()

with st.expander("Assumptions at a Glance", expanded=False):
    wa_pct = int(st.session_state["wa_demand_share"])
    nonwa_pct = 100 - wa_pct
    st.markdown(
        f"""
- Scenario: {scenario_name}
- Focus Year: {focus_year}
- Enrollment source weighting: {wa_pct}% WA demographic trend, {nonwa_pct}% non-resident pipeline trend
- UW enrollment trend (class size): {int(st.session_state["uw_enrollment_trend_pct"])}%
- Non-resident pipeline trend (by 2035): {int(st.session_state["national_trend_pct_by_2035"])}%
- Housing preference shift (by 2035): {int(st.session_state["behavior_headwind_pct_by_2035"])}%
- New capacity (net beds): {int(st.session_state["haggett_net_beds"])} (effective 2027)
- Annual rent increase: {float(st.session_state["rate_escalation_pct"]):.1f}%
- Annual cost growth: {float(st.session_state["expense_inflation_pct"]):.1f}%
- Debt service timing: {st.session_state["debt_shape"]}
        """.strip()
    )
    if st.session_state["debt_shape"] == "Custom":
        st.markdown(
            f"- Custom debt peak: {float(st.session_state['custom_peak_multiplier']):.2f}× starting {int(st.session_state['custom_peak_year'])}"
        )

tabs = st.tabs(["Dashboard", "Data Export"])


# -----------------------------
# Dashboard tab
# -----------------------------
with tabs[0]:
    # Demand / occupancy
    st.subheader("Demand: Beds Filled Over Time (Occupancy Index)")

    occ_df = df[["year", "Occupancy_Index", "Capacity_Index"]].copy()
    occ_df = occ_df.rename(
        columns={
            "Occupancy_Index": "Beds Filled (Occupancy Index)",
            "Capacity_Index": "Bed Capacity (Index)",
        }
    )
    occ_long = occ_df.melt("year", var_name="Series", value_name="Index")

    if HAS_ALTAIR:
        band = peak_band_layer(peak_window)

        y_dom = padded_domain(
            occ_long["Index"].to_numpy(dtype=float),
            pad_abs=5.0,
            force_include=(100.0, 100.0),
            clamp_low=0.0,
        )

        base = (
            alt.Chart(occ_long)
            .mark_line(point=False)
            .encode(
                x=x_year(),
                y=alt.Y("Index:Q", title="Index (2025 = 100)", scale=alt.Scale(domain=y_dom) if y_dom else alt.Undefined),
                color=alt.Color("Series:N", title=""),
                tooltip=[alt.Tooltip("year:Q", format="d"), "Series:N", alt.Tooltip("Index:Q", format=".1f")],
            )
            .properties(height=280)
        )

        ref_100 = alt.Chart(pd.DataFrame({"y": [100.0]})).mark_rule(color="#666", strokeDash=[4, 4]).encode(y="y:Q")

        layers = []
        if band is not None:
            layers.append(band)
        layers.extend([base, ref_100])

        chart = alt.layer(*layers).resolve_scale(x="shared", y="shared")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.line_chart(occ_df.set_index("year"))

    st.caption("If Beds Filled trends down while costs and debt pressure trend up, bond coverage becomes harder to maintain.")

    # NOI
    st.subheader("Operations: Buffer Available for Debt Service (NOI Index)")

    noi_df = df[["year", "NOI_Index"]].copy()
    noi_df = noi_df.rename(columns={"NOI_Index": "Operating Buffer (NOI Index)"})

    if HAS_ALTAIR:
        band = peak_band_layer(peak_window)

        y_dom = padded_domain(
            noi_df["Operating Buffer (NOI Index)"].to_numpy(dtype=float),
            pad_abs=10.0,
            force_include=(0.0, 100.0),
        )

        line = (
            alt.Chart(noi_df)
            .mark_line()
            .encode(
                x=x_year(),
                y=alt.Y(
                    "Operating Buffer (NOI Index):Q",
                    title="Index (100 = today’s buffer)",
                    scale=alt.Scale(domain=y_dom) if y_dom else alt.Undefined,
                ),
                tooltip=[alt.Tooltip("year:Q", format="d"), alt.Tooltip("Operating Buffer (NOI Index):Q", format=".1f")],
            )
            .properties(height=260)
        )

        ref_100 = alt.Chart(pd.DataFrame({"y": [100.0]})).mark_rule(color="#666", strokeDash=[4, 4]).encode(y="y:Q")
        ref_0 = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color="#888").encode(y="y:Q")

        layers = []
        if band is not None:
            layers.append(band)
        layers.extend([line, ref_100, ref_0])

        chart = alt.layer(*layers).resolve_scale(x="shared", y="shared")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.line_chart(noi_df.set_index("year"))

    st.caption("0 means operations generate no cash buffer for debt service. Below 0 means an operating loss (structural failure).")

    # Bond compliance
    st.subheader("Bond Compliance: Coverage Ratio vs Minimum Required")

    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("**Bond coverage ratio (DSCR) vs bond minimum requirement**")

        dscr_df = df[["year", "DSCR_Est"]].copy()
        dscr_df["Minimum Required (Bond Covenant)"] = required_dscr
        dscr_df = dscr_df.rename(columns={"DSCR_Est": "Coverage Ratio (DSCR)"})
        dscr_long = dscr_df.melt("year", var_name="Series", value_name="Value")

        if HAS_ALTAIR:
            band = peak_band_layer(peak_window)

            y_dom = padded_domain(
                dscr_long["Value"].to_numpy(dtype=float),
                pad_abs=0.25,
                force_include=(required_dscr, required_dscr),
                clamp_low=0.0,
            )

            lines = (
                alt.Chart(dscr_long)
                .mark_line()
                .encode(
                    x=x_year(),
                    y=alt.Y("Value:Q", title="Coverage ratio", scale=alt.Scale(domain=y_dom) if y_dom else alt.Undefined),
                    color=alt.Color(
                        "Series:N",
                        title="",
                        sort=["Coverage Ratio (DSCR)", "Minimum Required (Bond Covenant)"],
                    ),
                    strokeDash=alt.StrokeDash(
                        "Series:N",
                        title="",
                        sort=["Coverage Ratio (DSCR)", "Minimum Required (Bond Covenant)"],
                        legend=None,  # prevent duplicate legend blocks
                    ),
                    tooltip=[alt.Tooltip("year:Q", format="d"), "Series:N", alt.Tooltip("Value:Q", format=".2f")],
                )
                .properties(height=280)
            )

            layers = []
            if band is not None:
                layers.append(band)
            layers.append(lines)

            if breach_year is not None:
                breach_rule = (
                    alt.Chart(pd.DataFrame({"year": [breach_year]}))
                    .mark_rule(color="#b91c1c", strokeDash=[6, 3], opacity=0.9)
                    .encode(x=x_year(title="Year"))
                )
                layers.append(breach_rule)

            chart = alt.layer(*layers).resolve_scale(x="shared", y="shared")
            st.altair_chart(chart, use_container_width=True)
        else:
            st.line_chart(dscr_df.set_index("year")[["Coverage Ratio (DSCR)", "Minimum Required (Bond Covenant)"]])

        st.caption("If Coverage Ratio falls below the minimum line, it is a covenant breach.")

    with c_right:
        st.markdown("**Headroom above minimum (0 = breach)**")

        head_df = df[["year", "Headroom_Above_Min_DSCR"]].copy()
        head_df = head_df.rename(columns={"Headroom_Above_Min_DSCR": "Headroom (DSCR points)"})

        if HAS_ALTAIR:
            band = peak_band_layer(peak_window)

            y_dom = padded_domain(
                head_df["Headroom (DSCR points)"].to_numpy(dtype=float),
                pad_abs=0.25,
                force_include=(0.0, 0.0),
            )

            area_pos = (
                alt.Chart(head_df)
                .transform_filter(alt.datum["Headroom (DSCR points)"] >= 0)
                .mark_area(opacity=0.20, color="#16a34a")
                .encode(
                    x=x_year(),
                    y=alt.Y(
                        "Headroom (DSCR points):Q",
                        title="DSCR points above minimum",
                        scale=alt.Scale(domain=y_dom) if y_dom else alt.Undefined,
                    ),
                    y2=alt.value(0),
                )
            )

            area_neg = (
                alt.Chart(head_df)
                .transform_filter(alt.datum["Headroom (DSCR points)"] < 0)
                .mark_area(opacity=0.20, color="#dc2626")
                .encode(
                    x=x_year(),
                    y=alt.Y("Headroom (DSCR points):Q", title="DSCR points above minimum"),
                    y2=alt.value(0),
                )
            )

            line = (
                alt.Chart(head_df)
                .mark_line(color="#111827")
                .encode(
                    x=x_year(),
                    y=alt.Y(
                        "Headroom (DSCR points):Q",
                        title="DSCR points above minimum",
                        scale=alt.Scale(domain=y_dom) if y_dom else alt.Undefined,
                    ),
                    tooltip=[alt.Tooltip("year:Q", format="d"), alt.Tooltip("Headroom (DSCR points):Q", format="+.2f")],
                )
                .properties(height=280)
            )

            zero_rule = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color="#444").encode(y="y:Q")

            layers = []
            if band is not None:
                layers.append(band)
            layers.extend([area_pos, area_neg, line, zero_rule])

            if breach_year is not None:
                breach_rule = (
                    alt.Chart(pd.DataFrame({"year": [breach_year]}))
                    .mark_rule(color="#b91c1c", strokeDash=[6, 3], opacity=0.9)
                    .encode(x=x_year())
                )
                layers.append(breach_rule)

            chart = alt.layer(*layers).resolve_scale(x="shared", y="shared")
            st.altair_chart(chart, use_container_width=True)
        else:
            st.line_chart(head_df.set_index("year"))

        st.caption("This is the simplest ‘bond bill’ signal: above 0 means compliant, below 0 means breach.")

    with st.expander("Optional: Structural Balance (Revenue Index vs Cost Index)", expanded=False):
        sb_df = df[["year", "Revenue_Index", "Expense_Index"]].copy()
        sb_df = sb_df.rename(columns={"Revenue_Index": "Revenue Index", "Expense_Index": "Cost Index"})
        sb_long = sb_df.melt("year", var_name="Series", value_name="Index")

        if HAS_ALTAIR:
            band = peak_band_layer(peak_window)

            y_dom = padded_domain(
                sb_long["Index"].to_numpy(dtype=float),
                pad_abs=10.0,
                force_include=(100.0, 100.0),
                clamp_low=0.0,
            )

            lines = (
                alt.Chart(sb_long)
                .mark_line()
                .encode(
                    x=x_year(),
                    y=alt.Y("Index:Q", title="Index (2025 = 100)", scale=alt.Scale(domain=y_dom) if y_dom else alt.Undefined),
                    color=alt.Color("Series:N", title="", sort=["Revenue Index", "Cost Index"]),
                    tooltip=[alt.Tooltip("year:Q", format="d"), "Series:N", alt.Tooltip("Index:Q", format=".1f")],
                )
                .properties(height=260)
            )

            ref_100 = alt.Chart(pd.DataFrame({"y": [100.0]})).mark_rule(color="#666", strokeDash=[4, 4]).encode(y="y:Q")

            layers = []
            if band is not None:
                layers.append(band)
            layers.extend([lines, ref_100])

            chart = alt.layer(*layers).resolve_scale(x="shared", y="shared")
            st.altair_chart(chart, use_container_width=True)
        else:
            st.line_chart(sb_df.set_index("year"))

        st.caption("This is the classic ‘scissors’ view. The NOI Index chart above is usually the faster executive read.")


# -----------------------------
# Data export tab
# -----------------------------
with tabs[1]:
    st.subheader("Export Data (Indices Only, No Currency)")

    export_cols = [
        "year",
        "WA_18yo_Population",
        "WA_18yo_Index",
        "National_Global_Index",
        "Enrollment_Index",  # NEW
        "Behavior_Index",
        "Demographic_Index",
        "Demand_Index",
        "Capacity_Index",
        "Occupancy_Index",
        "Revenue_Index",
        "Expense_Index",
        "Debt_Index",
        "NOI_Index",
        "DSCR_Est",
        "Headroom_Above_Min_DSCR",
        "Safety_Margin_%",
        "Covenant_Breach",
    ]

    display_df = df[export_cols].copy()

    round_map = {c: 2 for c in export_cols if c not in {"year", "WA_18yo_Population", "Covenant_Breach"}}
    round_map.update(
        {
            "WA_18yo_Index": 1,
            "National_Global_Index": 1,
            "Enrollment_Index": 1,
            "Behavior_Index": 1,
            "Demographic_Index": 1,
            "Demand_Index": 1,
            "Capacity_Index": 1,
            "Occupancy_Index": 1,
            "Revenue_Index": 1,
            "Expense_Index": 1,
            "Debt_Index": 1,
            "NOI_Index": 1,
            "Safety_Margin_%": 1,
            "DSCR_Est": 2,
            "Headroom_Above_Min_DSCR": 2,
        }
    )

    display_df = display_df.round(round_map)

    st.dataframe(display_df, use_container_width=True, height=420)

    csv_buffer = StringIO()
    display_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download CSV (indices only)",
        data=csv_buffer.getvalue(),
        file_name="uw_hfs_housing_indices_only.csv",
        mime="text/csv",
    )

    st.caption("Export includes only indices and DSCR-derived metrics. No absolute currency values are generated or exported.")


# -----------------------------
# Sensitivity (tornado) intentionally disabled in v4.x
# -----------------------------
# Rationale: cognitive load > current value. Re-add when stakeholders ask for it explicitly.

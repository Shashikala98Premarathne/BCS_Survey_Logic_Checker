# ==========================================================
# Full BCS Survey Logic Checker
# ==========================================================
import re
import numpy as np
import pandas as pd
import streamlit as st
import io, csv, json
from io import BytesIO

# -------------------------------------------------------------------
# App setup
# -------------------------------------------------------------------
st.set_page_config(page_title="BCS Survey Logic Checker", layout="wide")
def set_background_solid(main="#6CD7E551", sidebar="#EEEFF3"):
    st.markdown(f"""
    <style>
      [data-testid="stAppViewContainer"],
      [data-testid="stAppViewContainer"] .main,
      [data-testid="stAppViewContainer"] .block-container {{
        background-color: {main} !important;
      }}
      [data-testid="stSidebar"],
      [data-testid="stSidebar"] > div,
      [data-testid="stSidebar"] .block-container {{
        background-color: {sidebar} !important;
      }}
      header[data-testid="stHeader"] {{ background: transparent; }}
      [data-testid="stDataFrame"],
      [data-testid="stTable"] {{ background-color: transparent !important; }}
    </style>
    """, unsafe_allow_html=True)
set_background_solid()
   
st.title("📊 BCS Survey Logic Checker")
st.caption("This tool is specifically designed for BCS Thailand/Taiwan. Identified mismatches will be highlighted in the deliverables..")

# -------------------------------------------------------------------
# File helpers
# -------------------------------------------------------------------
COMMON_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")

def _sniff_sep(sample_text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample_text[:4096], delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        return ","

def _norm_delim(sel: str) -> str:
    return {"\\t": "\t"}.get(sel, sel)

def read_any_table(uploaded_file, enc_override="auto", delim_override="auto", skip_bad=True) -> pd.DataFrame:
    name = (uploaded_file.name or "").lower()
    raw = uploaded_file.read()
    if raw.startswith(ZIP_SIGNATURES) or name.endswith((".xlsx", ".xls")):
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)

    encodings = COMMON_ENCODINGS if enc_override == "auto" else [enc_override]
    for enc_try in encodings:
        try:
            text = raw.decode(enc_try, errors="strict")
            sep = _sniff_sep(text) if delim_override == "auto" else _norm_delim(delim_override)
            kwargs = dict(encoding=enc_try, sep=sep, engine="python")
            if skip_bad:
                kwargs["on_bad_lines"] = "skip"
            return pd.read_csv(BytesIO(raw), **kwargs)
        except Exception:
            continue

    sep = "," if delim_override == "auto" else _norm_delim(delim_override)
    kwargs = dict(encoding="latin-1", sep=sep, engine="python")
    if skip_bad:
        kwargs["on_bad_lines"] = "skip"
    return pd.read_csv(BytesIO(raw), **kwargs)

# -------------------------------------------------------------------
# Sidebar upload
# -------------------------------------------------------------------
with st.sidebar:
    st.header("Input")
    data_file = st.file_uploader("Current wave data", type=["csv", "xlsx", "xls"])
    enc = st.selectbox("Encoding", ["auto", "utf-8", "utf-8-sig", "cp1252", "latin-1"], index=0)
    delim = st.selectbox("Delimiter", ["auto", ",", ";", "\\t", "|"], index=0)
    skip_bad = st.checkbox("Skip bad lines", value=True)

if not data_file:
    st.info("Upload a CSV/XLSX to begin.")
    st.stop()

try:
    data_file.seek(0)
    df = read_any_table(data_file, enc_override=enc, delim_override=delim, skip_bad=skip_bad)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

df.replace(
    {"#NULL!": np.nan, "NULL": np.nan, "null": np.nan, "NaN": np.nan, "nan": np.nan,
     "": np.nan, "na": np.nan, "N/A": np.nan, "n/a": np.nan},
    inplace=True,
)

# -------------------------------------------------------------------
# Rules list
# -------------------------------------------------------------------
SURVEY_RULES = {
    1: "Decision_maker=2 → terminate",
    2: "Fleet_knowledge=2 → terminate",
    3: "Company_position=98 → require OE",
    4: "HD trucks (S3) 0–99999, terminate if 0",
    5: "S3a sum must = S3",
    6: "Last purchase required if S3>0",
    7: "Main_brand autocode if single usage",
    8: "Quota_make must equal main_brand",
    9: "last_purchase_bX grid – only quota_make brand should have a value",
    10: "last_workshop_visit_bX grid – only quota_make brand should have a value",
    11: "last_workshop_visit_spareparts_bX grid – only quota_make brand should have a value",
    12: "Familiarity values 1–5, adjust if aware/usage",
    13: "Impression required if fam=2–5",
    14: "Preference auto if single consideration",
    15: "Quota satisfaction set required (E1, E4, E4b, E4c, F1, F2–F6)",
    16: "Truck_defects=1 → require OE",
    17: "Volvo quota → require satisfaction & dissatisfaction comments",
    18: "Barriers → require F9 follow-ups",
    19: "Transport_type=98 → require OE",
    20: "Volvo/Renault/Mack/Eicher quota → require operation_range_volvo_hdt",
    21: "Anonymity required for Volvo/Renault/Mack/Eicher",
    22: "System fields (region,country,survey_year) required"
}

digest, detailed = [], []

def add_issue(rule_id, msg, idx=None):
    digest.append((rule_id, msg))
    if idx is not None:
        detailed.append((idx, rule_id, msg))

# -------------------------------------------------------------------
# Checks
# -------------------------------------------------------------------
# Rule 1 – decision_maker
if "decision_maker" in df.columns:
    for i in df[df["decision_maker"]==2].index:
        add_issue(1,"decision_maker=2 (terminate)",i)

# Rule 2 – fleet_knowledge
if "fleet_knowledge" in df.columns:
    for i in df[df["fleet_knowledge"]==2].index:
        add_issue(2,"fleet_knowledge=2 (terminate)",i)

# Rule 3 – company_position=98 requires OE
if "company_position" in df.columns:
    if (df["company_position"]==98).any():
        if "company_position_other_specify" not in df.columns:
            for i in df[df["company_position"]==98].index:
                add_issue(3,"Missing OE for company_position=98",i)

# Rule 4 – HD trucks
if "n_heavy_duty_trucks" in df.columns:
    vals = pd.to_numeric(df["n_heavy_duty_trucks"], errors="coerce")
    for i in df[vals.isna()].index: add_issue(4,"Invalid S3 numeric",i)
    for i in df[(vals<0)|(vals>99999)].index: add_issue(4,"Out of range S3",i)
    for i in df[vals==0].index: add_issue(4,"S3=0 (terminate)",i)

# Rule 5 – S3a sum check
if set(["n_tractors","n_rigids","n_tippers","n_heavy_duty_trucks"]).issubset(df.columns):
    bad = df["n_heavy_duty_trucks"] != (df["n_tractors"]+df["n_rigids"]+df["n_tippers"])
    for i in df[bad].index: add_issue(5,"S3a1+2+3 != S3",i)

# Rule 6 – last_purchase_hdt required
if "last_purchase_hdt" not in df.columns:
    add_issue(6,"Missing last_purchase_hdt")

# Rule 7 – main_brand auto if single usage
usage_cols = [c for c in df.columns if c.startswith("usage_b")]
if "main_brand" in df.columns and usage_cols:
    one_brand = df[usage_cols].sum(axis=1)==1
    for i in df[one_brand & (df["main_brand"].isna())].index:
        add_issue(7,"main_brand should be auto from usage",i)

# Rule 8 – quota_make consistency
if "main_brand" in df.columns and "quota_make" in df.columns:
    bad = df["main_brand"]!=df["quota_make"]
    for i in df[bad].index: add_issue(8,"quota_make≠main_brand",i)

# Rule 9 – last_purchase grid: only quota brand should have a response
last_purch_cols = [c for c in df.columns if c.startswith("last_purchase_b")]
if last_purch_cols and "quota_make" in df.columns:
    for i, row in df.iterrows():
        qmake = str(row["quota_make"]).strip()
        if qmake in {"nan", "None", "", "NaN"}:
            add_issue(9, "Missing quota_make", i)
            continue

        quota_col = f"last_purchase_b{qmake}"
        if quota_col not in last_purch_cols:
            add_issue(9, f"No column found for quota_make={qmake}", i)
            continue

        # Check: quota brand cell must be filled
        if pd.isna(row.get(quota_col)):
            add_issue(9, f"Missing last_purchase for quota brand ({quota_col})", i)

        # Check: all other brand cells must be blank
        other_cols = [c for c in last_purch_cols if c != quota_col]
        for c in other_cols:
            if pd.notna(row.get(c)):
                add_issue(9, f"Non-quota brand {c} should be blank", i)


# Rule 10 – last_workshop_visit grid: only quota brand should have a response
workshop_cols = [c for c in df.columns if c.startswith("last_workshop_visit_b")]
if workshop_cols and "quota_make" in df.columns:
    for i, row in df.iterrows():
        qmake = str(row["quota_make"]).strip()
        if qmake in {"nan", "None", "", "NaN"}:
            add_issue(10, "Missing quota_make", i)
            continue

        quota_col = f"last_workshop_visit_b{qmake}"
        if quota_col not in workshop_cols:
            add_issue(10, f"No column found for quota_make={qmake}", i)
            continue

        # Quota brand cell must be filled
        if pd.isna(row.get(quota_col)):
            add_issue(10, f"Missing last_workshop_visit for quota brand ({quota_col})", i)

        # All other brands must be blank
        other_cols = [c for c in workshop_cols if c != quota_col]
        for c in other_cols:
            if pd.notna(row.get(c)):
                add_issue(10, f"Non-quota brand {c} should be blank", i)

# Rule 11 – last_workshop_visit_spareparts grid: only quota brand should have a response
spare_cols = [c for c in df.columns if c.startswith("last_workshop_visit_spareparts_b")]
if spare_cols and "quota_make" in df.columns:
    for i, row in df.iterrows():
        qmake = str(row["quota_make"]).strip()
        if qmake in {"nan", "None", "", "NaN"}:
            add_issue(11, "Missing quota_make", i)
            continue

        quota_col = f"last_workshop_visit_spareparts_b{qmake}"
        if quota_col not in spare_cols:
            add_issue(11, f"No column found for quota_make={qmake}", i)
            continue

        # Quota brand cell must be filled
        if pd.isna(row.get(quota_col)):
            add_issue(11, f"Missing last_workshop_visit_spareparts for quota brand ({quota_col})", i)

        # All other brands must be blank
        other_cols = [c for c in spare_cols if c != quota_col]
        for c in other_cols:
            if pd.notna(row.get(c)):
                add_issue(11, f"Non-quota brand {c} should be blank", i)


# Rule 12 – familiarity adjust
for col in [c for c in df.columns if c.startswith("familiarity_b")]:
    bid = col.split("_b")[-1]
    aware,usage = f"unaided_aware_b{bid}", f"usage_b{bid}"
    if aware in df.columns and usage in df.columns:
        bad = (df[aware]==1)|(df[usage]==1)
        for i in df[bad & (df[col]==1)].index:
            add_issue(12,f"{col}=1 despite awareness/usage",i)

# Rule 13 – impression required if fam=2–5
for col in [c for c in df.columns if c.startswith("familiarity_b")]:
    bid = col.split("_b")[-1]
    imp = f"overall_impression_b{bid}"
    if imp in df.columns:
        bad = df[col].isin([2,3,4,5]) & df[imp].isna()
        for i in df[bad].index: add_issue(13,f"{imp} missing where fam=2–5",i)

# Rule 14 – preference auto
cons_cols = [c for c in df.columns if c.startswith("consideration_b")]
if "preference" in df.columns and cons_cols:
    one = df[cons_cols].sum(axis=1)==1
    for i in df[one & df["preference"].isna()].index:
        add_issue(14,"preference should be auto from consideration",i)

# Rule 15 – quota satisfaction set
req = ["overall_satisfaction","likelihood_choose_brand","likelihood_choose_workshop","preference_strength","overall_rating_truck"]
for c in req:
    if c not in df.columns:
        add_issue(15,f"Missing {c}")

# Rule 16 – truck_defects
if "truck_defects" in df.columns and "truck_defects_other_specify" in df.columns:
    bad = (df["truck_defects"]==1)&df["truck_defects_other_specify"].isna()
    for i in df[bad].index: add_issue(16,"Missing OE for truck_defects=1",i)

# Rule 17 – Volvo comments
if "quota_make" in df.columns and (df["quota_make"]==38).any():
    for c in ["satisfaction_comments","dissatisfaction_comments"]:
        if c not in df.columns:
            for i in df[df["quota_make"]==38].index:
                add_issue(17,f"Missing {c} for Volvo",i)

# Rule 18 – barriers
if "reasons_not_consider_volvo" in df.columns:
    for follow in ["a_barriers_follow_up","b_barriers_follow_up","c_barriers_follow_up"]:
        if follow not in df.columns:
            for i in df.index: add_issue(18,f"Missing {follow}",i)

# Rule 19 – transport_type
if "transport_type" in df.columns and "transport_type_other_specify" in df.columns:
    bad = (df["transport_type"]==98)&df["transport_type_other_specify"].isna()
    for i in df[bad].index: add_issue(19,"Missing OE for transport_type=98",i)

# Rule 20 – operation_range
if "quota_make" in df.columns and "operation_range_volvo_hdt" in df.columns:
    bad = df["quota_make"].isin([38,31,23,9]) & df["operation_range_volvo_hdt"].isna()
    for i in df[bad].index: add_issue(20,"Missing operation_range_volvo_hdt",i)

# Rule 21 – anonymity
if "quota_make" in df.columns and "anonymity" in df.columns:
    bad = df["quota_make"].isin([38,31,23,9]) & df["anonymity"].isna()
    for i in df[bad].index: add_issue(21,"Missing anonymity",i)

# Rule 22 – system fields
for sysc in ["region","country","survey_year"]:
    if sysc not in df.columns:
        add_issue(22,f"Missing {sysc}")

# -------------------------------------------------------------------
# Outputs
# -------------------------------------------------------------------
digest_df = pd.DataFrame(digest,columns=["RuleID","Issue"]).drop_duplicates()
detailed_df = pd.DataFrame(detailed,columns=["RowID","RuleID","Issue"])

st.subheader("Survey Logic Issues")
if digest_df.empty:
    st.success("✅ No issues found – dataset follows survey logic.")
else:
    st.dataframe(digest_df, use_container_width=True)

    out = io.BytesIO()
    with pd.ExcelWriter(out,engine="xlsxwriter") as writer:
        digest_df.to_excel(writer,index=False,sheet_name="Digest")
        detailed_df.to_excel(writer,index=False,sheet_name="Detailed")
    st.download_button(
        "📥 Download Issues (Excel)",
        data=out.getvalue(),
        file_name="survey_logic_issues.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

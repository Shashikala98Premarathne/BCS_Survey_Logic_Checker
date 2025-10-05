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
    1:  "Decision_maker = 2 → terminate case",
    2:  "Fleet_knowledge = 2 → terminate case",
    3:  "Company_position = 98 → require company_position_other_specify",
    4:  "n_heavy_duty_trucks must be 0–99999; terminate if 0",
    #5:  "n_tractors / n_rigids / n_tippers must be valid (0–99999)",
    #6:  "Sum of n_tractors + n_rigids + n_tippers must equal n_heavy_duty_trucks",
    5:  "Missing last_purchase_hdt variable (required)",
    6:  "If only one brand used → main_brand should be auto-coded from usage",
    7:  "Quota_make must equal main_brand",
    8:  "last_purchase_bX grid – only quota_make brand should have a response (others blank)",
    9:  "last_workshop_visit_bX grid – only quota_make brand should have a response (others blank)",
    10: "last_workshop_visit_spareparts_bX grid – only quota_make brand should have a response (others blank)",
    11: "Familiarity = 1 invalid if brand aware/used",
    12: "If familiarity=2–5 → overall_impression_bX must be answered",
    13: "Consideration_bX grid – only quota_make brand should have a value (others blank)",
    14: "Preference should auto-fill if only one brand considered",
    15: "Performance should be blank if no brands considered",
    16: "Closeness_bX grid – should only be filled for considered brands (consideration_bX=1)",
    17: "Image_bX grid – should only be filled for brands that are aware, used, or have familiarity 2–5",
    18: "Image_31_bX should NOT exist (country-specific option)",
    19: "truck_defects=1 → require truck_defects_other_specify (OE)",
    20: "workshop_rating_14 should NOT exist (country-specific option)",
    21: "Quota_make=Volvo → require satisfaction_comments & dissatisfaction_comments",
    22: "If Volvo (38) NOT considered → reasons_not_consider_volvo required",
    23: "If Volvo (38) considered + F9 follow-ups if codes 3/4/10",
    24: "transport_type=98 → require transport_type_other_specify (OE)",
    25: "Volvo/Renault/Mack/Eicher quota → require operation_range_volvo_hdt",
    26: "Volvo/Renault/Mack/Eicher quota → require anonymity",
    27: "System fields region, country, and survey_year must exist",
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

# Rule 5 – Check S3a subtypes (tractors/rigids/tippers)
#needed_cols = ["n_tractors", "n_rigids", "n_tippers", "n_heavy_duty_trucks"]
#if set(needed_cols).issubset(df.columns):

#    for col in ["n_tractors", "n_rigids", "n_tippers"]:
#        vals = pd.to_numeric(df[col], errors="coerce")
#        # Range check 0–99999
#        out_of_range = (vals < 0) | (vals > 99999) | vals.isna()
#        for i in df[out_of_range].index:
#            add_issue(5, f"{col} out of range or invalid (must be 0–99999)", i)

    # Sum consistency check
    #vals_S3 = pd.to_numeric(df["n_heavy_duty_trucks"], errors="coerce")
    #vals_t = pd.to_numeric(df["n_tractors"], errors="coerce")
    #vals_r = pd.to_numeric(df["n_rigids"], errors="coerce")
    #vals_tp = pd.to_numeric(df["n_tippers"], errors="coerce")

    #bad_sum = vals_S3 != (vals_t.fillna(0) + vals_r.fillna(0) + vals_tp.fillna(0))
    #for i in df[bad_sum].index:
    #    add_issue(5, "Sum mismatch: n_tractors + n_rigids + n_tippers ≠ n_heavy_duty_trucks", i)


# Rule 5 – last_purchase_hdt required
if "last_purchase_hdt" not in df.columns:
    add_issue(5,"Missing last_purchase_hdt")

# Rule 6 – main_brand auto if single usage
usage_cols = [c for c in df.columns if c.startswith("usage_b")]
if "main_brand" in df.columns and usage_cols:
    one_brand = df[usage_cols].sum(axis=1)==1
    for i in df[one_brand & (df["main_brand"].isna())].index:
        add_issue(6,"main_brand should be auto from usage",i)

# Rule 7 – quota_make consistency
if "main_brand" in df.columns and "quota_make" in df.columns:
    bad = df["main_brand"]!=df["quota_make"]
    for i in df[bad].index: add_issue(7,"quota_make≠main_brand",i)

# Rule 8 – last_purchase grid: only quota brand should have a response
last_purch_cols = [c for c in df.columns if c.startswith("last_purchase_b")]
if last_purch_cols and "quota_make" in df.columns:
    for i, row in df.iterrows():
        qmake = str(row["quota_make"]).strip()
        if qmake in {"nan", "None", "", "NaN"}:
            add_issue(8, "Missing quota_make", i)
            continue

        quota_col = f"last_purchase_b{qmake}"
        if quota_col not in last_purch_cols:
            add_issue(8, f"No column found for quota_make={qmake}", i)
            continue

        # Check: quota brand cell must be filled
        if pd.isna(row.get(quota_col)):
            add_issue(8, f"Missing last_purchase for quota brand ({quota_col})", i)

        # Check: all other brand cells must be blank
        other_cols = [c for c in last_purch_cols if c != quota_col]
        for c in other_cols:
            if pd.notna(row.get(c)):
                add_issue(8, f"Non-quota brand {c} should be blank", i)


# Rule 9 – last_workshop_visit grid: only quota brand should have a response
workshop_cols = [c for c in df.columns if c.startswith("last_workshop_visit_b")]
if workshop_cols and "quota_make" in df.columns:
    for i, row in df.iterrows():
        qmake = str(row["quota_make"]).strip()
        if qmake in {"nan", "None", "", "NaN"}:
            add_issue(9, "Missing quota_make", i)
            continue

        quota_col = f"last_workshop_visit_b{qmake}"
        if quota_col not in workshop_cols:
            add_issue(9, f"No column found for quota_make={qmake}", i)
            continue

        # Quota brand cell must be filled
        if pd.isna(row.get(quota_col)):
            add_issue(9, f"Missing last_workshop_visit for quota brand ({quota_col})", i)

        # All other brands must be blank
        other_cols = [c for c in workshop_cols if c != quota_col]
        for c in other_cols:
            if pd.notna(row.get(c)):
                add_issue(9, f"Non-quota brand {c} should be blank", i)

# Rule 10 – last_workshop_visit_spareparts grid: only quota brand should have a response
spare_cols = [c for c in df.columns if c.startswith("last_workshop_visit_spareparts_b")]
if spare_cols and "quota_make" in df.columns:
    for i, row in df.iterrows():
        qmake = str(row["quota_make"]).strip()
        if qmake in {"nan", "None", "", "NaN"}:
            add_issue(10, "Missing quota_make", i)
            continue

        quota_col = f"last_workshop_visit_spareparts_b{qmake}"
        if quota_col not in spare_cols:
            add_issue(10, f"No column found for quota_make={qmake}", i)
            continue

        # Quota brand cell must be filled
        if pd.isna(row.get(quota_col)):
            add_issue(10, f"Missing last_workshop_visit_spareparts for quota brand ({quota_col})", i)

        # All other brands must be blank
        other_cols = [c for c in spare_cols if c != quota_col]
        for c in other_cols:
            if pd.notna(row.get(c)):
                add_issue(10, f"Non-quota brand {c} should be blank", i)


# Rule 11 – familiarity adjust
for col in [c for c in df.columns if c.startswith("familiarity_b")]:
    bid = col.split("_b")[-1]
    aware,usage = f"unaided_aware_b{bid}", f"usage_b{bid}"
    if aware in df.columns and usage in df.columns:
        bad = (df[aware]==1)|(df[usage]==1)
        for i in df[bad & (df[col]==1)].index:
            add_issue(11,f"{col}=1 despite awareness/usage",i)

# Rule 12 – impression required if fam=2–5
for col in [c for c in df.columns if c.startswith("familiarity_b")]:
    bid = col.split("_b")[-1]
    imp = f"overall_impression_b{bid}"
    if imp in df.columns:
        bad = df[col].isin([2,3,4,5]) & df[imp].isna()
        for i in df[bad].index: add_issue(12,f"{imp} missing where fam=2–5",i)

# Rule 13 – Consideration grid: should only be answered for quota make brand
cons_cols = [c for c in df.columns if c.startswith("consideration_b")]
if cons_cols and "quota_make" in df.columns:
    for i, row in df.iterrows():
        qmake = str(row["quota_make"]).strip()
        if qmake in {"nan", "None", "", "NaN"}:
            add_issue(13, "Missing quota_make", i)
            continue

        quota_col = f"consideration_b{qmake}"
        if quota_col not in cons_cols:
            add_issue(13, f"No consideration column found for quota_make={qmake}", i)
            continue

        # Quota brand must have a valid (non-blank) value
        if pd.isna(row.get(quota_col)) or str(row.get(quota_col)).strip() == "":
            add_issue(13, f"Missing consideration for quota brand ({quota_col})", i)

        # All other brand columns must be blank
        other_cols = [c for c in cons_cols if c != quota_col]
        for c in other_cols:
            val = row.get(c)
            if pd.notna(val) and str(val).strip() != "":
                add_issue(13, f"Non-quota brand {c} should be blank", i)

# Rule 14 – preference auto
cons_cols = [c for c in df.columns if c.startswith("consideration_b")]
if "preference" in df.columns and cons_cols:
    one = df[cons_cols].sum(axis=1)==1
    for i in df[one & df["preference"].isna()].index:
        add_issue(14,"preference should be auto from consideration",i)
        
# Rule 15 – Performance grid: should be blank if no brands considered
cons_cols = [c for c in df.columns if c.startswith("consideration_b")]
perf_cols = [c for c in df.columns if c.startswith("performance_b")]

if cons_cols and perf_cols:
    # Calculate if any brand considered at all (row-wise)
    any_considered = df[cons_cols].fillna(0).astype(float).sum(axis=1) > 0

    # Loop through each performance column (brand-specific)
    for pcol in perf_cols:
        m = re.search(r"_b(\d+)$", pcol)
        if not m:
            continue
        bid = m.group(1)
        perf = df[pcol]
        has_val = perf.notna() & (perf.astype(str).str.strip() != "")

        # Violation: no brands considered but performance filled
        bad = (~any_considered) & has_val
        for i in df[bad].index:
            add_issue(15, f"{pcol} should be blank (no brands considered in B3.a)", i)

# Rule 16 – Closeness grid: should only be filled for considered brands
cons_cols = [c for c in df.columns if c.startswith("consideration_b")]
close_cols = [c for c in df.columns if c.startswith("closeness_b")]

if cons_cols and close_cols:
    # Loop brand-by-brand
    for c in cons_cols:
        m = re.search(r"_b(\d+)$", c)
        if not m:
            continue
        bid = m.group(1)
        close_col = f"closeness_b{bid}"
        if close_col not in close_cols:
            continue

        # If brand not considered, closeness must be blank
        not_considered = (df[c].fillna(0).astype(float) != 1)
        has_closeness = df[close_col].notna() & (df[close_col].astype(str).str.strip() != "")
        bad = not_considered & has_closeness

        for i in df[bad].index:
            add_issue(16, f"{close_col} should be blank (brand not considered)", i)

# Rule 17 – Image grid: should only be filled for aware/usage/familiar brands
aware_cols = [c for c in df.columns if c.startswith("unaided_aware_b")]
usage_cols = [c for c in df.columns if c.startswith("usage_b")]
familiarity_cols = [c for c in df.columns if c.startswith("familiarity_b")]
image_cols = [c for c in df.columns if c.startswith("image_b")]

if image_cols:
    for img_col in image_cols:
        m = re.search(r"_b(\d+)$", img_col)
        if not m:
            continue
        bid = m.group(1)
        aware_col = f"unaided_aware_b{bid}"
        usage_col = f"usage_b{bid}"
        fam_col = f"familiarity_b{bid}"

        # Skip if brand not in any of those grids
        if all(c not in df.columns for c in [aware_col, usage_col, fam_col]):
            continue

        aware = df[aware_col] if aware_col in df.columns else 0
        usage = df[usage_col] if usage_col in df.columns else 0
        fam = df[fam_col] if fam_col in df.columns else np.nan

        allowed = (aware == 1) | (usage == 1) | (fam.isin([2, 3, 4, 5]))
        has_image = df[img_col].notna() & (df[img_col].astype(str).str.strip() != "")

        bad = (~allowed) & has_image
        for i in df[bad].index:
            add_issue(17, f"{img_col} should be blank (brand not aware/used/familiar)", i)

# Rule 18 – Image option 31 (country-specific) should not exist
bad_image_cols = [c for c in df.columns if c.lower().startswith("image_31_b")]
for c in bad_image_cols:
    add_issue(18, f"Column {c} should NOT exist (option 31 is country-specific)", i)

# Rule 19 – truck_defects
if "truck_defects" in df.columns and "truck_defects_other_specify" in df.columns:
    bad = (df["truck_defects"]==1)&df["truck_defects_other_specify"].isna()
    for i in df[bad].index: add_issue(19,"Missing OE for truck_defects=1",i)

# Rule 20 – workshop_rating_14 should NOT exist (country-specific)
bad_workshop_cols = [c for c in df.columns if c.lower().startswith("workshop_rating_14")]
for c in bad_workshop_cols:
    add_issue(20, f"Column {c} should NOT exist (option 14 is country-specific)", i)

# Rule 21 – Volvo comments
if "quota_make" in df.columns and (df["quota_make"]==38).any():
    for c in ["satisfaction_comments","dissatisfaction_comments"]:
        if c not in df.columns:
            for i in df[df["quota_make"]==38].index:
                add_issue(21,f"Missing {c} for Volvo",i)

# Rule 22 – Barriers: only if Volvo (38) NOT considered in B3.a
consid_col = "consideration_b38"
if consid_col in df.columns:
    for i, row in df.iterrows():
        considered_volvo = str(row.get(consid_col, "")).strip() in {"1", "1.0"}
        if considered_volvo:
            continue  # skip — Volvo considered

        # Volvo not considered → must have reasons_not_consider_volvo
        if "reasons_not_consider_volvo" not in df.columns:
            add_issue(22, "Missing reasons_not_consider_volvo column", i)
            continue

        reasons = row.get("reasons_not_consider_volvo")
        if pd.isna(reasons) or str(reasons).strip() == "":
            add_issue(22, "Missing reasons_not_consider_volvo (Volvo not considered)", i)
            continue

        # Normalize multi-select responses (comma, space, list, etc.)
        if isinstance(reasons, (list, set, tuple)):
            selected = {str(x).strip() for x in reasons}
        else:
            selected = {s.strip() for s in str(reasons).replace(";", ",").split(",") if s.strip()}

        # Check for follow-up conditions
        follow_map = {
            "3": "a_barriers_follow_up",
            "4": "b_barriers_follow_up",
            "10": "c_barriers_follow_up",
        }

        for code, follow_var in follow_map.items():
            if code in selected:
                if follow_var not in df.columns:
                    add_issue(23, f"Missing {follow_var} column", i)
                    continue
                if pd.isna(row.get(follow_var)) or str(row.get(follow_var)).strip() == "":
                    add_issue(23, f"Missing {follow_var} (required for code {code})", i)


# Rule 24 – transport_type
if "transport_type" in df.columns and "transport_type_other_specify" in df.columns:
    bad = (df["transport_type"]==98)&df["transport_type_other_specify"].isna()
    for i in df[bad].index: add_issue(24,"Missing OE for transport_type=98",i)

# Rule 25 – operation_range
if "quota_make" in df.columns and "operation_range_volvo_hdt" in df.columns:
    bad = df["quota_make"].isin([38,31,23,9]) & df["operation_range_volvo_hdt"].isna()
    for i in df[bad].index: add_issue(25,"Missing operation_range_volvo_hdt",i)

# Rule 26 – anonymity
if "quota_make" in df.columns and "anonymity" in df.columns:
    bad = df["quota_make"].isin([38,31,23,9]) & df["anonymity"].isna()
    for i in df[bad].index: add_issue(26,"Missing anonymity",i)

#Rule 27 – system fields
for sysc in ["region","country","survey_year"]:
    if sysc not in df.columns:
        add_issue(27,f"Missing {sysc}")

# -------------------------------------------------------------------
# Outputs
# -------------------------------------------------------------------
# Convert detailed issues to DataFrame
if detailed:
    results_df = pd.DataFrame(detailed, columns=["RowID", "RuleID", "Issue"])

    # Map rule descriptions
    results_df["Rule Description"] = results_df["RuleID"].map(SURVEY_RULES)

    # Add respondent ID if column exists in dataset
    if "respid" in df.columns:
        results_df["Respondent ID"] = results_df["RowID"].apply(
            lambda i: df.loc[i, "respid"] if i in df.index else np.nan
        )
    else:
        results_df["Respondent ID"] = np.nan

    # Reorder columns for readability
    results_df = results_df[
        ["Respondent ID", "RowID", "RuleID", "Rule Description", "Issue"]
    ]
else:
    results_df = pd.DataFrame(columns=["Respondent ID", "RowID", "RuleID", "Rule Description", "Issue"])


st.subheader("Survey Logic Issues")
if results_df.empty:
    st.success("✅ No issues found – dataset follows survey logic.")
else:
    st.dataframe(results_df, use_container_width=True)

from io import BytesIO

output = BytesIO()
with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    pd.DataFrame(digest, columns=["RuleID", "Issue"]).to_excel(writer, index=False, sheet_name="Digest")
    results_df.to_excel(writer, index=False, sheet_name="Detailed")
output.seek(0)

st.download_button(
    label="📥 Download Validation Report",
    data=output,
    file_name="BCS_Logic_Check_Report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

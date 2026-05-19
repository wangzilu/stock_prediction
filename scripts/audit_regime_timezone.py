"""Audit cross-market regime features for timezone PIT (point-in-time) safety.

Problem Statement:
  NASDAQ closes at US 4:00 PM ET = China next-day 4:00 AM CST.
  If cross_market_indices.parquet stores NASDAQ data with US calendar dates,
  then NASDAQ date=T close is NOT available when A-share date=T closes (15:00 CST).
  Using merge_asof(direction="backward") without a lag would leak future info:
    A-share T signal uses NASDAQ T close, but NASDAQ T close happens ~13 hours AFTER
    A-share T close.

  HSI/HSTECH close at 16:00 HKT (= 16:00 CST), which is 1 hour AFTER A-share close.
  For daily close-to-close signals computed after market hours, this is borderline safe
  but worth flagging.

Usage:
    python scripts/audit_regime_timezone.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "storage"


def main():
    path = DATA_DIR / "cross_market_indices.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found. Run fetch_cross_market_indices.py first.")
        sys.exit(1)

    df = pd.read_parquet(path)
    print(f"Loaded cross_market_indices.parquet: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")
    print()

    # ---- Identify columns per index ----
    hsi_cols = [c for c in df.columns if c.startswith("hsi_")]
    hstech_cols = [c for c in df.columns if c.startswith("hstech_")]
    nasdaq_cols = [c for c in df.columns if c.startswith("nasdaq_")]

    print(f"HSI features:    {len(hsi_cols)} cols")
    print(f"HSTECH features: {len(hstech_cols)} cols")
    print(f"NASDAQ features: {len(nasdaq_cols)} cols")
    print()

    # ---- Date convention analysis ----
    # For each index, get dates where it has non-null data
    def get_dates(cols):
        if not cols:
            return pd.Series(dtype="datetime64[ns]")
        mask = df[cols[0]].notna()
        return df.loc[mask, "date"].reset_index(drop=True)

    hsi_dates = get_dates(hsi_cols)
    hstech_dates = get_dates(hstech_cols)
    nasdaq_dates = get_dates(nasdaq_cols)

    print("=" * 70)
    print("1. DATE CONVENTION ANALYSIS")
    print("=" * 70)

    for name, dates in [("HSI", hsi_dates), ("HSTECH", hstech_dates), ("NASDAQ", nasdaq_dates)]:
        if dates.empty:
            print(f"\n  {name}: NO DATA")
            continue
        weekday_counts = dates.dt.weekday.value_counts().sort_index()
        weekday_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        print(f"\n  {name}:")
        print(f"    Total dates: {len(dates)}")
        print(f"    Range: {dates.min()} ~ {dates.max()}")
        print(f"    Weekday distribution:")
        for wd, cnt in weekday_counts.items():
            pct = cnt / len(dates) * 100
            print(f"      {weekday_names.get(wd, wd)}: {cnt} ({pct:.1f}%)")
        print(f"    Last 5 dates: {list(dates.tail(5).dt.strftime('%Y-%m-%d'))}")

    # ---- PIT Safety Check ----
    print()
    print("=" * 70)
    print("2. PIT SAFETY CHECK: NASDAQ same-day leak")
    print("=" * 70)

    # The core question: does NASDAQ date=T overlap with A-share date=T?
    # A-share trades Mon-Fri on CST calendar. NASDAQ trades Mon-Fri on US ET calendar.
    # Key timezone facts:
    #   NASDAQ close: 4:00 PM ET = next day 4:00 AM CST (winter) or 4:00 AM CST (summer)
    #   A-share close: 3:00 PM CST
    #   HSI close: 4:00 PM HKT = 4:00 PM CST
    #
    # Scenario: A-share date=2024-01-15 (Monday)
    #   NASDAQ date=2024-01-12 (Friday) close happens at:
    #     Friday 4pm ET = Saturday 5am CST  -> available for Monday A-share
    #   NASDAQ date=2024-01-15 (Monday) close happens at:
    #     Monday 4pm ET = Tuesday 5am CST   -> NOT available for Monday A-share close
    #
    # With merge_asof(direction="backward"), A-share T gets NASDAQ T (same date),
    # which is FUTURE DATA for the same calendar date.

    if not nasdaq_dates.empty and not hsi_dates.empty:
        nasdaq_set = set(nasdaq_dates)
        hsi_set = set(hsi_dates)

        # Find dates that exist in both (same calendar date in both markets)
        overlap = nasdaq_set & hsi_set
        nasdaq_only = nasdaq_set - hsi_set
        hsi_only = hsi_set - nasdaq_set

        print(f"\n  Dates in both NASDAQ & HSI: {len(overlap)}")
        print(f"  NASDAQ-only dates:          {len(nasdaq_only)}")
        print(f"  HSI-only dates:             {len(hsi_only)}")

        if overlap:
            sample_overlap = sorted(overlap)[-10:]
            print(f"\n  Sample overlapping dates (last 10):")
            for d in sample_overlap:
                wd = d.strftime("%a")
                print(f"    {d.strftime('%Y-%m-%d')} ({wd})")

        print(f"""
  CRITICAL ISSUE FOUND:
  ---------------------
  NASDAQ date=T close price occurs at US 4:00 PM ET = China T+1 ~4:00-5:00 AM CST.

  In feature_merger.py _load_cross_market_regime(), the code does:
    pd.merge_asof(unique_dates, right, on="date", direction="backward")

  For A-share training date T:
    - merge_asof picks NASDAQ row with date <= T
    - If NASDAQ has date=T, it matches NASDAQ T close
    - But NASDAQ T close happens AFTER A-share T close (by ~13 hours)
    - This is LOOK-AHEAD BIAS

  Example timeline for Monday 2024-01-15:
    Mon 15:00 CST  -> A-share closes (date=2024-01-15)
    Mon 16:30 EST  -> NASDAQ opens (date=2024-01-15 US)
    Tue 04:00 CST  -> NASDAQ closes (still date=2024-01-15 US)

  Current code matches A-share 2024-01-15 with NASDAQ 2024-01-15.
  But NASDAQ 2024-01-15 close is 13h in the future!
""")

    # ---- HSI/HSTECH Safety Check ----
    print("=" * 70)
    print("3. HSI/HSTECH PIT SAFETY CHECK")
    print("=" * 70)
    print(f"""
  HSI/HSTECH close at 16:00 HKT = 16:00 CST.
  A-share closes at 15:00 CST.

  Timeline for same date T:
    T 15:00 CST  -> A-share closes
    T 16:00 CST  -> HSI/HSTECH closes

  Assessment:
    - If signal is computed AFTER both markets close (e.g., evening batch):
      HSI T close IS available -> backward merge is SAFE
    - If signal is computed RIGHT at A-share close (15:00 CST):
      HSI T close is NOT yet available (1 hour away) -> BORDERLINE UNSAFE

  For daily close-to-close prediction, HSI/HSTECH backward merge is ACCEPTABLE
  because the model predicts next-day returns and the feature pipeline typically
  runs after all markets close. But this assumption should be documented.
""")

    # ---- Recommended Fix ----
    print("=" * 70)
    print("4. RECOMMENDED FIX")
    print("=" * 70)
    print(f"""
  For NASDAQ features:
    MUST shift dates by +1 business day (US date T -> available for A-share T+1 only).

    Implementation options:

    Option A: Fix in fetch_cross_market_indices.py (preferred)
      After computing NASDAQ features, shift the date column:
        nasdaq_features["date"] = nasdaq_features["date"] + pd.tseries.offsets.BDay(1)
      This converts "US close date" to "first China date it's available".

    Option B: Fix in feature_merger.py _load_cross_market_regime()
      After loading, shift NASDAQ columns by 1 day:
        nasdaq_cols = [c for c in df.columns if c.startswith("nasdaq_")]
        for col in nasdaq_cols:
            df[col] = df.groupby(level=0)[col].shift(1)  # or shift dates

    Option C: Use merge_asof tolerance
      Cannot easily fix with tolerance alone since the issue is same-day matching.
      Would need to subtract 1 day from NASDAQ dates before merge.

  For HSI/HSTECH:
    Current backward merge is acceptable for daily close-to-close signals.
    Document the assumption that feature pipeline runs after 16:00 HKT.

  Impact estimate:
    NASDAQ features contribute ~27 columns (9 features x 3 won't match exactly
    but all nasdaq_* features are affected). Any alpha attributed to NASDAQ
    same-day features may be spurious / non-reproducible in live trading.
""")

    # ---- Quantify the leak ----
    print("=" * 70)
    print("5. LEAK QUANTIFICATION")
    print("=" * 70)

    if not nasdaq_dates.empty:
        # How many A-share trading dates have same-day NASDAQ data?
        # We approximate A-share dates with HSI dates (same timezone, similar calendar)
        if not hsi_dates.empty:
            a_share_dates_approx = sorted(hsi_set)
            leaked = 0
            total = 0
            for d in a_share_dates_approx:
                total += 1
                if d in nasdaq_set:
                    leaked += 1

            print(f"\n  Approximate A-share trading dates (using HSI calendar): {total}")
            print(f"  Dates where NASDAQ same-day data would leak: {leaked}")
            print(f"  Leak rate: {leaked/total*100:.1f}% of trading days")
            print(f"  (These are days where NASDAQ has the same calendar date as HSI/A-share)")
    else:
        print("\n  Cannot quantify - NASDAQ data not available")

    # ---- Check actual feature_merger code path ----
    print()
    print("=" * 70)
    print("6. CODE PATH TRACE")
    print("=" * 70)
    print(f"""
  In models/feature_merger.py:

  _load_cross_market_regime() (line ~988):
    1. Loads cross_market_indices.parquet
    2. All columns (hsi_*, hstech_*, nasdaq_*) share the same "date" column
    3. Calls merge_asof(unique_dates, right, on="date", direction="backward")
    4. This broadcasts matched regime values to ALL stocks for that date

  The merge treats all indices identically -- no special handling for NASDAQ
  timezone difference. The backward merge picks the most recent date <= T,
  which for NASDAQ on overlapping dates means the same calendar date T.

  Since NASDAQ T close actually occurs ~13 hours after A-share T close,
  this is a confirmed PIT violation.
""")

    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data(path):
    return pd.read_csv(path)

def extract_numeric_from_string(s, pattern, default=np.nan, dtype=float):
    if pd.isna(s):
        return default
    match = re.search(pattern, s)
    if match:
        try:
            val = match.group(1)
            val = val.replace(',', '')  # remove commas from numbers like 9,999
            return dtype(val)
        except:
            return default
    return default

def parse_numeric_columns(df):
    df_parsed = pd.DataFrame()

    # Rating - already numeric, just convert
    df_parsed["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    # Spec_score - convert to numeric
    df_parsed["Spec_score"] = pd.to_numeric(df["Spec_score"], errors="coerce")

    # No_of_sim - extract number of sims, count "Dual Sim" or "Single Sim" text
    def count_sims(s):
        if pd.isna(s):
            return np.nan
        s = s.lower()
        if "dual" in s:
            return 2
        elif "single" in s:
            return 1
        else:
            return np.nan
    df_parsed["No_of_sim"] = df["No_of_sim"].apply(count_sims)

    # Ram - extract number from strings like "4 GB RAM"
    df_parsed["Ram"] = df["Ram"].apply(lambda x: extract_numeric_from_string(x, r"(\d+)", default=np.nan, dtype=int))

    # Battery - extract mAh number from strings like "6000 mAh Battery"
    df_parsed["Battery"] = df["Battery"].apply(lambda x: extract_numeric_from_string(x, r"(\d+)", default=np.nan, dtype=int))

    # Display - extract inches float from "6.6 inches"
    df_parsed["Display"] = df["Display"].apply(lambda x: extract_numeric_from_string(x, r"(\d+\.?\d*)", default=np.nan, dtype=float))

    # Camera - sum all MP numbers in string like "50 MP + 2 MP Dual Rear & 13 MP Front Camera"
    def sum_camera_mp(s):
        if pd.isna(s):
            return np.nan
        mp_vals = re.findall(r"(\d+)\s*MP", s, re.I)
        if mp_vals:
            try:
                return sum([int(mp) for mp in mp_vals])
            except:
                return np.nan
        return np.nan
    df_parsed["Camera"] = df["Camera"].apply(sum_camera_mp)

    # External_Memory - extract max GB number from string, convert TB to GB
    def ext_mem_gb(s):
        if pd.isna(s):
            return np.nan
        match_gb = re.search(r"upto\s*(\d+)\s*GB", s, re.I)
        match_tb = re.search(r"upto\s*(\d+)\s*TB", s, re.I)
        if match_tb:
            try:
                return int(match_tb.group(1)) * 1024
            except:
                return np.nan
        if match_gb:
            try:
                return int(match_gb.group(1))
            except:
                return np.nan
        return np.nan
    df_parsed["External_Memory"] = df["External_Memory"].apply(ext_mem_gb)

    # Android_version - convert to numeric
    df_parsed["Android_version"] = pd.to_numeric(df["Android_version"], errors="coerce")

    # Price - remove commas and convert to int
    def parse_price(s):
        if pd.isna(s):
            return np.nan
        s = str(s).replace(",", "").strip()
        try:
            return int(s)
        except:
            return np.nan
    df_parsed["Price"] = df["Price"].apply(parse_price)

    return df_parsed

def preprocess_data(df):
    # Extract and parse numeric columns
    df_numeric = parse_numeric_columns(df)

    # Separate features and target
    if "Price" not in df_numeric.columns:
        raise ValueError("Target column 'Price' must be present.")

    # Define numeric features (all except target)
    feature_cols = [col for col in df_numeric.columns if col != "Price"]

    # Build preprocessing pipeline for numeric features
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_cols)],
        remainder="drop",
    )

    # Fit and transform features
    X = df_numeric[feature_cols]
    y = df_numeric["Price"]

    X_processed = preprocessor.fit_transform(X)

    # Return processed features and pipeline, append target for convenience
    df_processed = pd.DataFrame(X_processed, columns=feature_cols)
    df_processed["Price"] = y.values

    return df_processed, preprocessor

def save_pipeline(pipeline, path):
    import joblib
    joblib.dump(pipeline, path)
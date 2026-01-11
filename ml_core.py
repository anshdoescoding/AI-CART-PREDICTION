# ml_core.py
import pandas as pd
import re
from collections import defaultdict, Counter
from typing import List, Tuple, Optional

def normalize_name(name: str) -> str:
    """Standardize product names for matching."""
    if pd.isna(name):
        return ""
    name = str(name).lower().strip()
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name

def _build_product_catalog(products_df: pd.DataFrame, product_col="product", mrp_col="mrp"):
    """
    Build product -> avg MRP mapping and canonical product name list from given products_df.
    Expects one product per row in products_df (catalog or transactional dataset).
    """
    if products_df is None or products_df.empty:
        return {}, []

    # normalize product names
    products_df = products_df.copy()
    products_df["_prod_norm"] = products_df[product_col].apply(normalize_name)

    # average MRP per normalized product name
    mrp_map = products_df.groupby("_prod_norm")[mrp_col].agg(lambda s: pd.to_numeric(s, errors="coerce").dropna().mean())
    mrp_map = mrp_map.to_dict()

    all_products = sorted([p for p in mrp_map.keys() if p])
    return mrp_map, all_products

def _build_cooccurrence_matrix(df_transactions: pd.DataFrame, cart_col: str):
    """
    Build co-occurrence counts between normalized product names from the transaction DataFrame.
    df_transactions: has column cart_col which contains comma-separated product names (or list)
    Returns dict-of-dict counts: cooccur[a][b] = times a and b appeared together
    """
    cooccur = defaultdict(Counter)

    def parse_cart(x):
        if pd.isna(x):
            return []
        if isinstance(x, list):
            items = x
        else:
            items = [i.strip() for i in str(x).split(",") if i.strip()]
        items = [normalize_name(i) for i in items if i.strip()]
        # unique within one transaction
        return list(dict.fromkeys(items))

    for _, row in df_transactions.iterrows():
        items = parse_cart(row.get(cart_col, ""))
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a and b:
                    cooccur[a][b] += 1
                    cooccur[b][a] += 1
    return cooccur

def predict_products(cart_items: List[str], cooccur: dict, all_products: List[str], n: int = 1) -> List[str]:
    """
    Predict top-n products most likely to appear given cart_items (based on cooccurrence).
    Excludes products already in cart_items.
    """
    scores = Counter()
    cart_norm = [normalize_name(x) for x in cart_items if x]
    for item in cart_norm:
        if item in cooccur:
            for other, cnt in cooccur[item].items():
                if other not in cart_norm:
                    scores[other] += cnt
    # fall back to overall most frequent products if no cooccurrence info
    if not scores:
        # simply pick most common products from all_products excluding cart_norm
        candidates = [p for p in all_products if p not in cart_norm]
        return candidates[:n]
    preds = [p for p, _ in scores.most_common() if p not in cart_norm]
    return preds[:n]

def recommend_products(cart_items: List[str], cooccur: dict, all_products: List[str],
                       exclude: List[str] = None, n: int = 3) -> List[str]:
    """
    Recommend additional products (distinct from predictions and cart).
    Strategy: rank by cooccurrence score but ensure no overlap with `exclude`.
    """
    exclude = set(normalize_name(x) for x in (exclude or []))
    cart_norm = set(normalize_name(x) for x in cart_items if x)
    scores = Counter()

    for item in cart_norm:
        if item in cooccur:
            for other, cnt in cooccur[item].items():
                if other not in cart_norm and other not in exclude:
                    scores[other] += cnt

    if not scores:
        # fallback: top products in catalog excluding cart and exclude
        candidates = [p for p in all_products if p not in cart_norm and p not in exclude]
        return candidates[:n]

    recs = [p for p, _ in scores.most_common() if p not in cart_norm and p not in exclude]
    # ensure we return exactly n (or as many as possible)
    additional = [p for p in all_products if p not in cart_norm and p not in exclude and p not in recs]
    recs.extend(additional)
    return recs[:n]

def calc_recommendation_mrp(recommendations: List[str], mrp_map: dict) -> Tuple[List[float], float]:
    """
    For each recommended product (normalized name) return its average MRP (or 0 if unknown),
    and total MRP sum.
    """
    mrps = []
    for prod in recommendations:
        prod_norm = normalize_name(prod)
        mrp = float(mrp_map.get(prod_norm, 0.0))
        mrps.append(round(mrp, 2))
    total = round(sum(mrps), 2)
    return mrps, total

def run_pipeline(input_df: pd.DataFrame,
                 cart_col_name: str = "Products in cart",
                 products_df: Optional[pd.DataFrame] = None,
                 product_col_in_products_df: str = "product",
                 mrp_col_in_products_df: str = "mrp",
                 topk_prediction: int = 1,
                 topk_recommendation: int = 3) -> pd.DataFrame:
    """
    Main entry point.
    - input_df: uploaded transactional DataFrame which includes a cart column (comma-separated product names)
    - products_df: a catalog/transactional dataset that contains product names and their MRP (used to compute avg MRPs)
      If None, the function will try to use input_df as the product source (if it has product/mrp cols).
    Returns input_df with appended columns:
      - Product Prediction (comma separated normalized product names)
      - Product Recommendation (comma separated)
      - Recommendation MRP (comma separated MRPs per recommended item)
      - Recommendation MRP Sum (float)
    """
    df = input_df.copy()

    if cart_col_name not in df.columns:
        raise ValueError(f"Cart column '{cart_col_name}' not found in input DataFrame")

    # If products_df wasn't provided, attempt to infer it from input_df columns
    if products_df is None:
        if product_col_in_products_df in df.columns and mrp_col_in_products_df in df.columns:
            products_df = df[[product_col_in_products_df, mrp_col_in_products_df]].copy()
        else:
            # If dataset_filled.csv exists in current folder and has required columns, try loading it.
            try:
                fallback = pd.read_csv("dataset_filled.csv")
                if product_col_in_products_df in fallback.columns and mrp_col_in_products_df in fallback.columns:
                    products_df = fallback[[product_col_in_products_df, mrp_col_in_products_df]].copy()
            except Exception:
                products_df = pd.DataFrame(columns=[product_col_in_products_df, mrp_col_in_products_df])

    mrp_map, all_products = _build_product_catalog(products_df, product_col=product_col_in_products_df, mrp_col=mrp_col_in_products_df)

    # Build cooccurrence from the input dataframe (transactions) — use cart column there.
    cooccur = _build_cooccurrence_matrix(df, cart_col_name)

    # create a parsed list column for convenience
    def parse_cart_items(x):
        if pd.isna(x):
            return []
        if isinstance(x, list):
            items = x
        else:
            items = [i.strip() for i in str(x).split(",") if i.strip()]
        return [normalize_name(i) for i in items if i.strip()]

    df["_CartList"] = df[cart_col_name].apply(parse_cart_items)

    # compute predictions and recommendations
    preds = []
    recs = []
    rec_mrps = []
    rec_mrp_sums = []
    for cart in df["_CartList"]:
        p = predict_products(cart, cooccur, all_products, n=topk_prediction)
        # ensure predictions are normalized strings
        p = [normalize_name(x) for x in p]

        # exclude predictions + cart from recommendations
        exclude = set(p) | set(normalize_name(x) for x in cart)
        r = recommend_products(cart, cooccur, all_products, exclude=exclude, n=topk_recommendation)
        r = [normalize_name(x) for x in r]

        mrp_list, mrp_sum = calc_recommendation_mrp(r, mrp_map)

        preds.append(", ".join(p))
        recs.append(", ".join(r))
        rec_mrps.append(", ".join(str(x) for x in mrp_list))
        rec_mrp_sums.append(mrp_sum)

    df["Product Prediction"] = preds
    df["Product Recommendation"] = recs
    df["Recommendation MRP"] = rec_mrps
    df["Recommendation MRP Sum"] = rec_mrp_sums

    # clean up helper column
    df.drop(columns=["_CartList"], inplace=True, errors="ignore")
    return df

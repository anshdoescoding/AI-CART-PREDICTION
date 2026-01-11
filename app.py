# app.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, os

from flask import Flask, render_template, request, send_from_directory
import pandas as pd
from ml_core import run_pipeline

app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route("/process", methods=["POST"])
def process_file():
    file = request.files.get("file")
    if not file:
        return "No file uploaded", 400

    # Try reading uploaded file (Excel or CSV)
    df = None
    file.seek(0)
    try:
        try:
            df = pd.read_excel(file, engine="openpyxl")
        except Exception:
            file.seek(0)
            # pandas detect separator if sep=None with engine='python'
            df = pd.read_csv(file, engine="python", on_bad_lines="skip")
    except Exception as e:
        return f"File parsing failed: {e}", 400

    if df is None or df.empty:
        return "Uploaded file is empty or not a valid Excel/CSV", 400

    # Detect cart column
    possible_cart_cols = [
        "products in cart", "cart products", "cart items", "items in cart",
        "cart", "product list", "products"
    ]
    cart_col = None
    for candidate in possible_cart_cols:
        for col in df.columns:
            if candidate in col.lower():
                cart_col = col
                break
        if cart_col:
            break

    if not cart_col:
        return f"Pipeline failed: No cart column found. Columns detected: {list(df.columns)}", 400

    # Attempt to load product catalog (dataset_filled.csv) from working directory
    products_df = None
    catalog_path = "dataset_filled.csv"
    if os.path.exists(catalog_path):
        try:
            products_df = pd.read_csv(catalog_path)
        except Exception:
            products_df = None

    # Run pipeline
    try:
        df_out = run_pipeline(df, cart_col_name=cart_col, products_df=products_df,
                              product_col_in_products_df="product", mrp_col_in_products_df="mrp",
                              topk_prediction=1, topk_recommendation=3)
    except Exception as e:
        return f"Pipeline failed while running model: {e}", 500

    # Save output CSV to static for download
    os.makedirs(app.static_folder, exist_ok=True)
    output_path = os.path.join(app.static_folder, "output.csv")
    df_out.to_csv(output_path, index=False)

    # Build top recommendation plot
    all_recs = []
    if "Product Recommendation" in df_out.columns:
        for r in df_out["Product Recommendation"].dropna():
            all_recs.extend([x.strip() for x in str(r).split(",") if x.strip()])

    plot_url = None
    if all_recs:
        rec_series = pd.Series(all_recs).value_counts().head(10)
        plt.figure(figsize=(8, 5))
        plt.barh(rec_series.index[::-1], rec_series.values[::-1])
        plt.title("Top 10 Recommended Products")
        plt.xlabel("Frequency")
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format="png", bbox_inches="tight")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

    preview_html = df_out.head(10).to_html(classes="table table-striped", index=False, escape=False)

    return render_template(
        "results.html",
        tables=preview_html,
        plot_url=plot_url,
        download_link=os.path.join("static", "output.csv")
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)

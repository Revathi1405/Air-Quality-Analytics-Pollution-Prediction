import csv
import math
import random
from collections import Counter
from statistics import mean, median

import streamlit as st

st.set_page_config(page_title="Air Quality Analytics", page_icon="🌿", layout="wide")

DATA_PATH = "data/air_quality_data.csv"
TARGET = "CO(GT)"
FEATURES = ["PT08.S1(CO)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "T", "RH", "AH"]


def to_float(value):
    try:
        x = float(value)
        return None if x == -200 else x
    except Exception:
        return None


def load_data(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def clean_rows(rows):
    cleaned = []
    for row in rows:
        vals = {k: to_float(row.get(k, "")) for k in FEATURES + [TARGET]}
        if vals[TARGET] is None:
            continue
        cleaned.append({"Date": row["Date"], "Time": row["Time"], **vals})
    # Mean imputation for missing sensor values
    for col in FEATURES:
        valid = [r[col] for r in cleaned if r[col] is not None]
        fill = mean(valid) if valid else 0.0
        for r in cleaned:
            if r[col] is None:
                r[col] = fill
    return cleaned


def split_data(rows, test_ratio=0.2, seed=42):
    data = rows[:]
    random.Random(seed).shuffle(data)
    cut = int(len(data) * (1 - test_ratio))
    return data[:cut], data[cut:]


def simple_linear_fit(train, feature):
    xs = [r[feature] for r in train]
    ys = [r[TARGET] for r in train]
    mx, my = mean(xs), mean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else 0.0
    intercept = my - slope * mx
    return intercept, slope


def linear_predict(model, x):
    intercept, slope = model
    return intercept + slope * x


def distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def feature_scales(train, features):
    scales = {}
    for f in features:
        vals = [r[f] for r in train]
        m = mean(vals)
        sd = math.sqrt(mean([(v - m) ** 2 for v in vals])) or 1.0
        scales[f] = (m, sd)
    return scales


def vector(row, features, scales):
    return [(row[f] - scales[f][0]) / scales[f][1] for f in features]


def knn_predict(train, row, features, scales, k=7):
    q = vector(row, features, scales)
    distances = []
    for r in train:
        distances.append((distance(q, vector(r, features, scales)), r[TARGET]))
    distances.sort(key=lambda z: z[0])
    neighbors = distances[:k]
    return sum(y for _, y in neighbors) / len(neighbors)


def metrics(actual, predicted):
    n = len(actual) or 1
    mae = sum(abs(a - p) for a, p in zip(actual, predicted)) / n
    rmse = math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / n)
    ybar = sum(actual) / n
    ss_tot = sum((a - ybar) ** 2 for a in actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return mae, rmse, r2


def html_bar_chart(items, title, width=680):
    if not items:
        return ""
    maxv = max(v for _, v in items) or 1
    bars = []
    for label, value in items:
        pct = max(4, min(100, value / maxv * 100))
        bars.append(
            f"<div style='margin:10px 0'><div style='display:flex;justify-content:space-between;font-size:13px'><span>{label}</span><b>{value:.2f}</b></div>"
            f"<div style='background:#e8eef3;border-radius:8px;height:14px'><div style='width:{pct:.1f}%;background:#2e7d32;height:14px;border-radius:8px'></div></div></div>"
        )
    return f"<div style='max-width:{width}px'><h4>{title}</h4>{''.join(bars)}</div>"


rows = load_data(DATA_PATH)
cleaned = clean_rows(rows)

st.title("🌿 Air Quality Analytics & Pollution Prediction")
st.caption("Data Analytics and Visualization Mini Project | Environmental Domain")

# Train models once per app session
@st.cache_data(show_spinner=False)
def build_models(data):
    train, test = split_data(data)
    lr_feature = "PT08.S1(CO)"
    lr = simple_linear_fit(train, lr_feature)
    lr_pred = [linear_predict(lr, r[lr_feature]) for r in test]
    lr_metrics = metrics([r[TARGET] for r in test], lr_pred)

    knn_features = FEATURES
    scales = feature_scales(train, knn_features)
    # Evaluate on a smaller test sample for a responsive dashboard.
    eval_test = test[:300]
    knn_pred = [knn_predict(train, r, knn_features, scales, k=7) for r in eval_test]
    knn_metrics = metrics([r[TARGET] for r in eval_test], knn_pred)
    return train, test, lr, scales, lr_metrics, knn_metrics

train, test, lr_model, scales, lr_metrics, knn_metrics = build_models(cleaned)

with st.sidebar:
    st.header("Project Information")
    st.write("**Domain:** Environment / Nature")
    st.write(f"**Records:** {len(cleaned):,}")
    st.write(f"**Training records:** {len(train):,}")
    st.write(f"**Testing records:** {len(test):,}")
    st.write("**Target:** CO(GT)")
    st.info("Missing sensor values (-200) are replaced using mean imputation.")


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Pollution Analysis", "Environmental Analysis", "ML Comparison", "Prediction"
])

with tab1:
    st.subheader("Air Quality Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(rows):,}")
    c2.metric("Usable Records", f"{len(cleaned):,}")
    c3.metric("Training Records", f"{len(train):,}")
    c4.metric("Test Records", f"{len(test):,}")

    avg_co = mean(r[TARGET] for r in cleaned)
    avg_temp = mean(r["T"] for r in cleaned)
    avg_rh = mean(r["RH"] for r in cleaned)
    st.markdown("### Key Environmental Indicators")
    a, b, c = st.columns(3)
    a.metric("Average CO", f"{avg_co:.2f}")
    b.metric("Average Temperature", f"{avg_temp:.2f} °C")
    c.metric("Average Humidity", f"{avg_rh:.2f}%")

    sample = cleaned[:10]
    st.markdown("### Sample of Processed Data")
    st.table([{k: r[k] for k in ["Date", "Time", TARGET, "T", "RH", "NOx(GT)"]} for r in sample])

with tab2:
    st.subheader("Pollution Analysis")
    st.markdown(html_bar_chart([
        ("CO(GT)", mean(r[TARGET] for r in cleaned)),
        ("NOx(GT)", mean(r["NOx(GT)"] for r in cleaned)),
        ("NO2(GT)", mean(r["NO2(GT)"] for r in cleaned)),
    ], "Average Pollution Measurements"), unsafe_allow_html=True)

    # Ten-point CO trend, created from ordered records without external plotting libraries.
    step = max(1, len(cleaned) // 10)
    trend = []
    for i in range(0, len(cleaned), step):
        chunk = cleaned[i:i + step]
        if chunk:
            trend.append((str(i + 1), mean(r[TARGET] for r in chunk)))
    st.markdown("### CO Trend Across the Dataset")
    st.table([{"Segment": label, "Average CO": round(value, 3)} for label, value in trend[:10]])

with tab3:
    st.subheader("Temperature, Humidity and Air Quality")
    st.markdown(html_bar_chart([
        ("Temperature (°C)", avg_temp),
        ("Relative Humidity (%)", avg_rh),
        ("Absolute Humidity", mean(r["AH"] for r in cleaned)),
    ], "Environmental Averages"), unsafe_allow_html=True)

    correlations = []
    y = [r[TARGET] for r in cleaned]
    my = mean(y)
    for f in FEATURES:
        x = [r[f] for r in cleaned]
        mx = mean(x)
        den = math.sqrt(sum((v - mx) ** 2 for v in x) * sum((v - my) ** 2 for v in y))
        corr = sum((a - mx) * (b - my) for a, b in zip(x, y)) / den if den else 0
        correlations.append((f, corr))
    correlations.sort(key=lambda z: abs(z[1]), reverse=True)
    st.markdown("### Feature Relationship with CO")
    st.table([{"Feature": f, "Correlation with CO": round(c, 3)} for f, c in correlations])

with tab4:
    st.subheader("Machine Learning Model Comparison")
    st.write("Two regression approaches are implemented and evaluated on held-out data.")
    comparison = [
        {"Model": "Linear Regression", "MAE": lr_metrics[0], "RMSE": lr_metrics[1], "R²": lr_metrics[2]},
        {"Model": "KNN Regression", "MAE": knn_metrics[0], "RMSE": knn_metrics[1], "R²": knn_metrics[2]},
    ]
    st.table([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()} for row in comparison])
    best = min(comparison, key=lambda x: x["RMSE"])
    st.success(f"Best model by RMSE: {best['Model']}")
    st.markdown("**Linear Regression:** predicts CO from the CO sensor response (PT08.S1(CO)).")
    st.markdown("**KNN Regression:** predicts CO from multiple environmental and pollution features using standardized distance.")

with tab5:
    st.subheader("CO Pollution Prediction")
    st.write("Enter sensor/environment values to compare predictions from both models.")

    col1, col2 = st.columns(2)
    with col1:
        s1 = st.number_input("PT08.S1(CO)", value=float(mean(r["PT08.S1(CO)"] for r in cleaned)))
        c6h6 = st.number_input("C6H6(GT)", value=float(mean(r["C6H6(GT)"] for r in cleaned)))
        nox = st.number_input("NOx(GT)", value=float(mean(r["NOx(GT)"] for r in cleaned)))
        no2 = st.number_input("NO2(GT)", value=float(mean(r["NO2(GT)"] for r in cleaned)))
    with col2:
        temp = st.number_input("Temperature (T)", value=float(mean(r["T"] for r in cleaned)))
        rh = st.number_input("Relative Humidity (RH)", value=float(mean(r["RH"] for r in cleaned)))
        ah = st.number_input("Absolute Humidity (AH)", value=float(mean(r["AH"] for r in cleaned)))

    if st.button("Predict CO Level", type="primary"):
        user_row = {
            "PT08.S1(CO)": s1, "C6H6(GT)": c6h6, "NOx(GT)": nox,
            "NO2(GT)": no2, "T": temp, "RH": rh, "AH": ah
        }
        lr_value = linear_predict(lr_model, s1)
        knn_value = knn_predict(train, user_row, FEATURES, scales, k=7)
        p1, p2 = st.columns(2)
        p1.metric("Linear Regression Prediction", f"{max(0, lr_value):.3f}")
        p2.metric("KNN Regression Prediction", f"{max(0, knn_value):.3f}")
        st.info("Predictions are for academic demonstration using the supplied project dataset.")

st.markdown("---")
st.caption("Air Quality Analytics Project | Preprocessing → EDA → Visualization → ML Comparison → Prediction")

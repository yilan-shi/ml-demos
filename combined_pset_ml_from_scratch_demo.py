import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import os

st.set_page_config(page_title="ML from Scratch", layout="wide", page_icon="🧬")

# ── core implementations (no sklearn for logistic regression) ─────────────────
#
# Three distinct training functions, because the original PSET files use three
# different setups. Each is kept faithful to the corresponding submitted file
# rather than unified, so the demo reflects the actual work.

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logreg(X, y, lr=0.0001, steps=1000):
    """Logistic regression via gradient ASCENT on log-likelihood, pure NumPy.
    Mirrors PSET7_2.py / PSET7_3_3.py: un-normalized gradient, bias appended.
    Update rule: w += lr * Xᵀ(y − σ(Xw))"""
    X_b = np.hstack([X, np.ones((X.shape[0], 1))])
    w = np.zeros(X_b.shape[1])
    lls = []
    for _ in range(steps):
        h = sigmoid(X_b @ w)
        w += lr * (X_b.T @ (y - h))
        ll = np.mean(y * np.log(h + 1e-10) + (1 - y) * np.log(1 - h + 1e-10))
        lls.append(ll)
    return w, lls

def train_logreg_ancestry(X, y, lr=0.0001, steps=1000):
    """Logistic regression mirroring PSET7_3_23andme_ancestry.py exactly:
    bias column appended via hstack, NORMALIZED gradient (divided by len(y)),
    gradient DESCENT on (preds - y).
    Update rule: w -= lr * Xᵀ(σ(Xw) − y) / n
    The log-likelihood list is recorded only for any plotting — it does not
    affect the weights or accuracy, and is not part of the original file."""
    X_b = np.hstack([X, np.ones((X.shape[0], 1))])
    w = np.zeros(X_b.shape[1])
    lls = []
    for _ in range(steps):
        h = sigmoid(X_b @ w)
        grad = X_b.T @ (h - y) / len(y)
        w -= lr * grad
        ll = np.mean(y * np.log(h + 1e-10) + (1 - y) * np.log(1 - h + 1e-10))
        lls.append(ll)
    return w, lls

def train_logreg_heart(X, y, lr, steps=1000):
    """Logistic regression mirroring PSET7_4.py exactly: NO bias column,
    NORMALIZED gradient (divided by sample count), gradient DESCENT on (h - y).
    Update rule: w -= lr * Xᵀ(σ(Xw) − y) / n
    The log-likelihood list is recorded only for the convergence plot — it does
    not affect the weights or accuracy, and is not part of the original PSET7_4.py."""
    w = np.zeros(X.shape[1])
    y_col = y.reshape(-1, 1) if y.ndim == 1 else y
    w = w.reshape(-1, 1)
    lls = []
    for _ in range(steps):
        h = sigmoid(X @ w)
        gradient = X.T @ (h - y_col) / y_col.size
        w -= lr * gradient
        ll = np.mean(y_col * np.log(h + 1e-10) + (1 - y_col) * np.log(1 - h + 1e-10))
        lls.append(ll)
    return w.flatten(), lls

def predict_proba(X, w):
    """Prediction for the bias-appended models (Tab 1, Tab 2)."""
    X_b = np.hstack([X, np.ones((X.shape[0], 1))])
    return sigmoid(X_b @ w)

def predict_proba_nobias(X, w):
    """Prediction for the PSET7_4.py-style model: no bias column appended."""
    return sigmoid(X @ w)

def acc(probas, y_true):
    return np.mean((probas >= 0.5).astype(int) == y_true)

DATA = "data"

def load(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        st.error(f"Missing file: {path}. Place CSV files in the data/ folder.")
        st.stop()
    return pd.read_csv(path)

# ── palette ───────────────────────────────────────────────────────────────────

BLUE   = "#4c60f0"
DBLUE  = "#1a2fa8"
LBLUE  = "#b8c1ff"
RED    = "#e24b4a"
GRAY   = "#888"

def style_fig(fig):
    fig.patch.set_facecolor("none")
    for ax in fig.axes:
        ax.set_facecolor("none")
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["bottom","left"]].set_color("#cccccc")
        ax.tick_params(colors="#555", labelsize=9)
        ax.xaxis.label.set_color("#555")
        ax.yaxis.label.set_color("#555")
        ax.title.set_color("#222")
    return fig

# ── header ─────────────────────────────────────────────────────────────────────

st.title("Machine Learning from Scratch")
st.caption(
    "Logistic regression, calibration, and linear regression — "
    "implemented in across five datasets. From prior CS hw assignment (S2025 quarter)"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "① Logistic Regression",
    "② Ancestry Classifier",
    "③ Heart Disease",
    "④ Calibration",
    "⑤ Caltrain Regression",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — simple logistic regression  (mirrors PSET7_2.py)
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Logistic Regression from Scratch")
    st.markdown(
        "Gradient ascent on log-likelihood, implemented in NumPy.  \n"
        "Update rule: `w += lr · Xᵀ(y − σ(Xw))`"
    )

    train1 = load("simple-train.csv")
    test1  = load("simple-test.csv")
    X_tr1  = train1.drop("Label", axis=1).values
    y_tr1  = train1["Label"].values.astype(float)
    X_te1  = test1.drop("Label", axis=1).values
    y_te1  = test1["Label"].values.astype(float)

    c_ctrl, c_plot = st.columns([1, 2])

    with c_ctrl:
        lr1    = st.select_slider("Learning rate", [0.0001, 0.001, 0.01, 0.1], 0.0001, key="lr1")
        steps1 = st.slider("Training steps", 100, 3000, 1000, 100, key="s1")

    w1, lls1 = train_logreg(X_tr1, y_tr1, lr=lr1, steps=steps1)
    p1 = predict_proba(X_te1, w1)

    with c_ctrl:
        st.metric("Test accuracy", f"{acc(p1, y_te1):.1%}")
        st.metric("w₁  (x1)", f"{w1[0]:.5f}")
        st.metric("w₂  (x2)", f"{w1[1]:.5f}")
        st.metric("Bias", f"{w1[2]:.5f}")

    with c_plot:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

        axes[0].plot(lls1, color=BLUE, linewidth=1.5)
        axes[0].set_xlabel("Step"); axes[0].set_ylabel("Log-likelihood")
        axes[0].set_title("Training convergence"); axes[0].grid(alpha=.2)

        xs = np.linspace(-0.3, 1.3, 200)
        if abs(w1[1]) > 1e-8:
            boundary = -(w1[0] * xs + w1[2]) / w1[1]
            axes[1].plot(xs, boundary, "k-", lw=1.4, label="Decision boundary")
        clrs = [RED if y == 0 else BLUE for y in y_te1]
        axes[1].scatter(X_te1[:, 0], X_te1[:, 1], c=clrs, s=80, zorder=5,
                        edgecolors="white", lw=0.5)
        axes[1].set_xlabel("x1"); axes[1].set_ylabel("x2")
        axes[1].set_title("Test set + decision boundary")
        axes[1].legend(fontsize=9); axes[1].grid(alpha=.2)

        plt.tight_layout()
        st.pyplot(style_fig(fig)); plt.close()

    with st.expander("View raw data"):
        d1, d2 = st.columns(2)
        d1.caption("Train"); d1.dataframe(train1, use_container_width=True)
        d2.caption("Test");  d2.dataframe(test1,  use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ancestry classifier  (mirrors PSET7_3_23andme_ancestry.py)
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Ancestry Classifier")
    st.markdown(
        "Logistic regression on 20 binary SNP (single-nucleotide polymorphism) "
        "features — the same structure 23andMe uses for ancestry prediction.  \n"
        "Normalized gradient descent: `w -= lr · Xᵀ(σ(Xw) − y) / n`"
    )

    train2 = load("ancestry-train.csv")
    test2  = load("ancestry-test.csv")
    X_tr2  = train2.drop("Label", axis=1).values
    y_tr2  = train2["Label"].values.astype(float)
    X_te2  = test2.drop("Label", axis=1).values
    y_te2  = test2["Label"].values.astype(float)
    feat2  = [c for c in train2.columns if c != "Label"]

    c_ctrl2, c_plot2 = st.columns([1, 2])

    with c_ctrl2:
        lr2    = st.select_slider("Learning rate", [0.00001, 0.0001, 0.001], 0.0001, key="lr2")
        steps2 = st.slider("Training steps", 500, 5000, 1000, 500, key="s2")

    w2, _ = train_logreg_ancestry(X_tr2, y_tr2, lr=lr2, steps=steps2)
    p2    = predict_proba(X_te2, w2)
    fw2   = w2[:-1]
    best_snp = feat2[np.argmax(np.abs(fw2))]

    with c_ctrl2:
        st.metric("Test accuracy", f"{acc(p2, y_te2):.1%}")
        st.metric("Most predictive SNP", best_snp)
        st.metric("Its weight", f"{fw2[np.argmax(np.abs(fw2))]:.5f}")
        st.metric("Bias term", f"{w2[-1]:.5f}")

    with c_plot2:
        fig2, axes2 = plt.subplots(1, 2, figsize=(9, 3.5))

        si = np.argsort(np.abs(fw2))[::-1][:10]
        colors_bar = [BLUE if fw2[i] > 0 else RED for i in si]
        axes2[0].barh([feat2[i] for i in si], [fw2[i] for i in si], color=colors_bar)
        axes2[0].axvline(0, color="black", lw=0.5)
        axes2[0].set_xlabel("Weight"); axes2[0].set_title("Top 10 SNP weights")
        axes2[0].grid(alpha=.2, axis="x")

        axes2[1].hist(p2[y_te2 == 0], bins=20, alpha=0.6, color=RED,    density=True, label="Class 0")
        axes2[1].hist(p2[y_te2 == 1], bins=20, alpha=0.6, color=BLUE,   density=True, label="Class 1")
        axes2[1].axvline(0.5, color="black", ls="--", lw=1, label="Threshold 0.5")
        axes2[1].set_xlabel("Predicted probability"); axes2[1].set_ylabel("Density")
        axes2[1].set_title("Score distribution by class")
        axes2[1].legend(fontsize=9); axes2[1].grid(alpha=.2)

        plt.tight_layout()
        st.pyplot(style_fig(fig2)); plt.close()

    st.divider()
    st.subheader("Try a prediction")
    st.caption("Enter a 20-bit SNP profile (comma-separated 0s and 1s)")
    snp_in = st.text_input("SNP values", "1,0,0,0,0,0,0,1,1,1,0,1,0,1,0,0,1,0,1,1")
    try:
        snp_arr = np.array([float(x.strip()) for x in snp_in.split(",")])
        if len(snp_arr) == 20:
            prob2 = predict_proba(snp_arr.reshape(1, -1), w2)[0]
            st.metric("P(Class 1)", f"{prob2:.4f}")
            st.progress(float(prob2))
        else:
            st.warning(f"Need 20 values, got {len(snp_arr)}")
    except Exception:
        st.error("Enter 20 comma-separated 0/1 values")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — heart disease: learning rate search  (mirrors PSET7_4.py)
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Heart Disease Classifier — Learning Rate Search")
    st.markdown(
        "22-feature medical dataset. Grid search over 8 learning rates; "
        "best selected by test accuracy.  \n"
        "No bias column; normalized gradient descent: `w -= η · Xᵀ(σ(Xw) − y) / n`"
    )

    train3 = load("heart-train.csv")
    test3  = load("heart-test.csv")
    X_tr3  = train3.iloc[:, :-1].values
    y_tr3  = train3.iloc[:,  -1].values.astype(float)
    X_te3  = test3.iloc[:, :-1].values
    y_te3  = test3.iloc[:,  -1].values.astype(float)

    lrs3 = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    rows3 = []
    for eta in lrs3:
        w3, _ = train_logreg_heart(X_tr3, y_tr3, lr=eta, steps=1000)
        rows3.append({"η": eta, "Test accuracy": acc(predict_proba_nobias(X_te3, w3), y_te3)})
    df3 = pd.DataFrame(rows3)
    best3 = df3.loc[df3["Test accuracy"].idxmax()]

    c_ctrl3, c_plot3 = st.columns([1, 2])

    with c_ctrl3:
        st.metric("Best η", str(best3["η"]))
        st.metric("Best accuracy", f"{best3['Test accuracy']:.1%}")
        st.metric("Training samples", len(train3))
        st.metric("Features", X_tr3.shape[1])
        st.dataframe(
            df3.style.highlight_max(subset=["Test accuracy"], color=LBLUE)
                     .format({"η": "{}", "Test accuracy": "{:.3f}"}),
            use_container_width=True
        )

    with c_plot3:
        fig3, axes3 = plt.subplots(1, 2, figsize=(9, 3.5))

        bar_colors = [DBLUE if eta == best3["η"] else LBLUE for eta in lrs3]
        axes3[0].bar([str(r) for r in lrs3], df3["Test accuracy"], color=bar_colors)
        axes3[0].set_xlabel("Learning rate η"); axes3[0].set_ylabel("Accuracy")
        axes3[0].set_title("Grid search results"); axes3[0].set_ylim(0, 1)
        axes3[0].grid(alpha=.2, axis="y")

        _, lls_best3  = train_logreg_heart(X_tr3, y_tr3, lr=best3["η"], steps=1000)
        worst3_eta = lrs3[df3["Test accuracy"].values.argmin()]
        _, lls_worst3 = train_logreg_heart(X_tr3, y_tr3, lr=worst3_eta, steps=1000)
        axes3[1].plot(lls_best3,  color=DBLUE, lw=1.5, label=f"Best η={best3['η']}")
        axes3[1].plot(lls_worst3, color=RED,   lw=1,   ls="--", alpha=.7, label=f"Worst η={worst3_eta}")
        axes3[1].set_xlabel("Step"); axes3[1].set_ylabel("Log-likelihood")
        axes3[1].set_title("Convergence: best vs worst η")
        axes3[1].legend(fontsize=9); axes3[1].grid(alpha=.2)

        plt.tight_layout()
        st.pyplot(style_fig(fig3)); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — probability calibration  (mirrors PSET7_5.py)
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Probability Calibration")
    st.markdown(
        "A calibrated model predicts 60% probability only when ~60% of those examples "
        "are truly positive. This measures whether the logistic regression output "
        "is a trustworthy probability."
    )

    cal4 = load("ancestry-calibration.csv")
    buckets4 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def nearest_bucket(p):
        return min(buckets4, key=lambda x: abs(x - p))

    cal4["bucket"] = cal4["LogRegPr"].apply(nearest_bucket)

    rows4 = []
    for b in buckets4:
        sub = cal4[cal4["bucket"] == b]
        if len(sub):
            rows4.append({"Predicted bucket": b, "Empirical rate": sub["Label"].mean(), "N": len(sub)})
    df4 = pd.DataFrame(rows4)

    c_ctrl4, c_plot4 = st.columns([1, 2])

    with c_ctrl4:
        st.dataframe(
            df4.style.format({"Predicted bucket": "{:.1f}", "Empirical rate": "{:.5f}", "N": "{}"}),
            use_container_width=True
        )
        b06 = df4[df4["Predicted bucket"] == 0.6]
        if not b06.empty:
            st.metric("Bucket 0.6 empirical rate", f"{b06['Empirical rate'].values[0]:.5f}")

    with c_plot4:
        fig4, ax4 = plt.subplots(figsize=(5, 4.5))
        ax4.plot([0, 1], [0, 1], "k--", lw=1, alpha=.5, label="Perfect calibration")
        ax4.scatter(df4["Predicted bucket"], df4["Empirical rate"],
                    s=df4["N"] * 2.5, c=BLUE, alpha=.85, zorder=5,
                    edgecolors="white", lw=0.5)
        ax4.plot(df4["Predicted bucket"], df4["Empirical rate"],
                 color=BLUE, lw=1.5, label="Our model")
        ax4.set_xlabel("Predicted probability (bucket)")
        ax4.set_ylabel("Empirical positive rate")
        ax4.set_title("Calibration plot\n(bubble size = samples in bucket)")
        ax4.legend(fontsize=9); ax4.grid(alpha=.2)
        ax4.set_xlim(-0.05, 1.05); ax4.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        st.pyplot(style_fig(fig4)); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Caltrain: linear regression + MLE residual inference
#         (mirrors PSET7_7a.py + PSET7_7b.py)
# ═══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.header("Caltrain Ridership — Linear Regression + Probabilistic Error")
    st.markdown(
        "Predict passengers per hour from 7 features. Then: estimate residual "
        "variance with MLE and compute P(model is wrong by > 20 passengers)."
    )

    train5 = load("caltrain-train.csv")
    test5  = load("caltrain-test.csv")

    feats5 = ["is_summer", "is_weekend", "is_holiday", "busy_time_of_day",
              "north_or_southbound", "temperature", "chance_of_rain"]

    X_tr5 = train5[feats5].values; y_tr5 = train5["passengers_per_hour"].values
    X_te5 = test5[feats5].values;  y_te5 = test5["passengers_per_hour"].values

    m5 = LinearRegression().fit(X_tr5, y_tr5)
    pred5 = m5.predict(X_te5)
    resid5 = y_te5 - pred5
    rmse5  = np.sqrt(mean_squared_error(y_te5, pred5))
    sigma5 = np.sqrt(np.mean(resid5 ** 2))
    p_off  = 2 * (1 - norm.cdf(20 / sigma5))

    c_ctrl5, c_plot5 = st.columns([1, 2])

    with c_ctrl5:
        st.metric("RMSE", f"{rmse5:.2f} passengers")
        st.metric("MLE σ² (residual variance)", f"{np.mean(resid5**2):.2f}")
        st.metric("MLE σ", f"{sigma5:.2f}")
        st.metric("P(|error| > 20 passengers)", f"{p_off:.6f}")
        coef5 = pd.DataFrame({"Feature": feats5, "Coefficient": m5.coef_})
        coef5 = coef5.reindex(coef5["Coefficient"].abs().sort_values(ascending=False).index)
        st.dataframe(coef5.style.format({"Coefficient": "{:.4f}"}), use_container_width=True)

    with c_plot5:
        fig5, axes5 = plt.subplots(1, 2, figsize=(9, 3.5))

        axes5[0].scatter(y_te5, pred5, alpha=.35, color=BLUE, s=12, edgecolors="none")
        lim = [min(y_te5.min(), pred5.min()), max(y_te5.max(), pred5.max())]
        axes5[0].plot(lim, lim, "k--", lw=1, label="y = ŷ")
        axes5[0].set_xlabel("Actual passengers/hr"); axes5[0].set_ylabel("Predicted passengers/hr")
        axes5[0].set_title("Predicted vs actual"); axes5[0].legend(fontsize=9)
        axes5[0].grid(alpha=.2)

        xr = np.linspace(resid5.min() - 10, resid5.max() + 10, 300)
        axes5[1].hist(resid5, bins=40, density=True, color=LBLUE, edgecolor="none", label="Residuals")
        axes5[1].plot(xr, norm.pdf(xr, 0, sigma5), color=DBLUE, lw=2, label=f"N(0, σ={sigma5:.1f})")
        axes5[1].axvline( 20, color=RED, ls="--", lw=1.2, label="±20 threshold")
        axes5[1].axvline(-20, color=RED, ls="--", lw=1.2)
        axes5[1].set_xlabel("Residual (actual − predicted)"); axes5[1].set_ylabel("Density")
        axes5[1].set_title(f"Residuals + Gaussian fit\nP(|err|>20) = {p_off:.4f}")
        axes5[1].legend(fontsize=8); axes5[1].grid(alpha=.2)

        plt.tight_layout()
        st.pyplot(style_fig(fig5)); plt.close()

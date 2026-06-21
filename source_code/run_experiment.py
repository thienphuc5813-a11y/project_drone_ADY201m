"""
run_experiment.py
=================
File thực thi chính – kết nối toàn bộ pipeline:

  dataset.py        →  Nạp & xử lý dữ liệu
  models.py         →  Định nghĩa 5 mô hình
  xai_evaluator.py  →  Tính XAI Fidelity (SHAP + Kendall-Tau)

Output (lưu vào thư mục `results/`):
  summary_mae.csv              – Bảng MAE 5 mô hình
  summary_fidelity.csv         – Bảng Kendall-Tau 5 mô hình
  01_mae_comparison.png        – Bar chart MAE
  02_xai_fidelity.png          – Bar chart Kendall-Tau
  03_shap_summary_best_model.png  – SHAP summary plot
  04_feature_importance_vs_groundtruth.png  – Grouped bar chart
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # không cần màn hình (headless)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import mean_absolute_error

# ── Import 3 module nội bộ ────────────────────────────────────────────────────
from dataset       import load_and_preprocess, split_and_scale
from models        import get_models
from xai_evaluator import (
    build_soft_ground_truth,
    extract_shap_importance,
    compute_fidelity_scores,
)

# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình chung
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH    = "real_data/flights_ready_for_eda.csv"
RESULTS_DIR  = "results"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# Palette học thuật
PALETTE_5 = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

# Thiết lập style matplotlib học thuật
plt.rcParams.update({
    "figure.dpi"       : 150,
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
    "axes.titlesize"   : 14,
    "axes.labelsize"   : 12,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "grid.linestyle"   : "--",
    "legend.framealpha": 0.8,
    "savefig.bbox": "tight",
})

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Bước 0: Chuẩn bị thư mục output
# ─────────────────────────────────────────────────────────────────────────────
def setup_results_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    print(f"[run] Thư mục output: '{path}/'")


# ─────────────────────────────────────────────────────────────────────────────
# Bước 1: Nạp & xử lý dữ liệu
# ─────────────────────────────────────────────────────────────────────────────
def step1_load_data():
    print("\n" + "="*60)
    print("BƯỚC 1: Nạp và xử lý dữ liệu")
    print("="*60)
    df = load_and_preprocess(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, scaler, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# Bước 2: Huấn luyện và đánh giá MAE
# ─────────────────────────────────────────────────────────────────────────────
def step2_train_and_evaluate(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("BƯỚC 2: Huấn luyện 5 mô hình & tính MAE")
    print("="*60)

    models     = get_models(random_state=RANDOM_STATE)
    mae_scores = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"\n  → Đang huấn luyện: {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae    = mean_absolute_error(y_test, y_pred)
        mae_scores[name]    = mae
        fitted_models[name] = model
        print(f"     MAE = {mae:.4f} W")

    # In bảng tóm tắt
    print("\n  ── Bảng tóm tắt MAE ──")
    print(f"  {'Model':>25s}  {'MAE (W)':>10s}")
    print("  " + "-"*38)
    for name, mae in sorted(mae_scores.items(), key=lambda x: x[1]):
        marker = " ← tốt nhất" if mae == min(mae_scores.values()) else ""
        print(f"  {name:>25s}  {mae:>10.4f}{marker}")

    return fitted_models, mae_scores


# ─────────────────────────────────────────────────────────────────────────────
# Bước 3: XAI – tính Ground Truth và SHAP Importance
# ─────────────────────────────────────────────────────────────────────────────
def step3_xai(fitted_models, X_train, X_test, feature_names):
    print("\n" + "="*60)
    print("BƯỚC 3: Phân tích XAI (SHAP + Ground Truth)")
    print("="*60)

    # 3a. Soft Ground Truth từ Ridge
    print("\n  ── 3a. Tạo Soft Ground Truth (Ridge |coef_|) ──")
    ground_truth = build_soft_ground_truth(X_train, y_train, feature_names)

    # 3b. SHAP cho từng mô hình
    print("\n  ── 3b. Trích xuất SHAP Importance ──")
    shap_importances = {}
    shap_values_dict = {}

    for name, model in fitted_models.items():
        imp, sv = extract_shap_importance(
            model_name    = name,
            fitted_model  = model,
            X_train       = X_train,
            X_test        = X_test,
            feature_names = feature_names,
            random_state  = RANDOM_STATE,
        )
        shap_importances[name] = imp
        shap_values_dict[name] = sv

    # 3c. Tính Fidelity (Kendall-Tau)
    print("\n  ── 3c. Tính Fidelity Scores (Kendall-Tau) ──")
    fidelity_scores = compute_fidelity_scores(shap_importances, ground_truth)

    return ground_truth, shap_importances, shap_values_dict, fidelity_scores


# ─────────────────────────────────────────────────────────────────────────────
# Bước 4: Lưu bảng kết quả CSV
# ─────────────────────────────────────────────────────────────────────────────
def step4_save_tables(mae_scores, fidelity_scores):
    print("\n" + "="*60)
    print("BƯỚC 4: Lưu bảng kết quả CSV")
    print("="*60)

    # summary_mae.csv
    df_mae = pd.DataFrame(
        list(mae_scores.items()), columns=["Model", "MAE_Watts"]
    ).sort_values("MAE_Watts").reset_index(drop=True)
    df_mae["Rank"] = range(1, len(df_mae) + 1)
    path_mae = os.path.join(RESULTS_DIR, "summary_mae.csv")
    df_mae.to_csv(path_mae, index=False)
    print(f"  Lưu: {path_mae}")

    # summary_fidelity.csv
    df_fid = pd.DataFrame(
        list(fidelity_scores.items()), columns=["Model", "Kendall_Tau"]
    ).sort_values("Kendall_Tau", ascending=False).reset_index(drop=True)
    df_fid["Rank"] = range(1, len(df_fid) + 1)
    path_fid = os.path.join(RESULTS_DIR, "summary_fidelity.csv")
    df_fid.to_csv(path_fid, index=False)
    print(f"  Lưu: {path_fid}")

    return df_mae, df_fid


# ─────────────────────────────────────────────────────────────────────────────
# Bước 5: Trực quan hóa
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, filename: str) -> None:
    """Lưu figure và đóng."""
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Lưu: {path}")


def plot_01_mae_comparison(mae_scores: dict) -> None:
    """Bar chart so sánh MAE 5 mô hình."""
    names  = list(mae_scores.keys())
    values = [mae_scores[n] for n in names]
    order  = np.argsort(values)          # sắp xếp tăng dần

    names_sorted  = [names[i]  for i in order]
    values_sorted = [values[i] for i in order]
    colors        = [PALETTE_5[i % len(PALETTE_5)] for i in range(len(names_sorted))]
    # Màu đặc biệt cho mô hình tốt nhất
    colors[0] = "#F44336"

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names_sorted, values_sorted, color=colors, edgecolor="white",
                   linewidth=0.8, height=0.55)

    # Nhãn giá trị trên từng bar
    for bar, val in zip(bars, values_sorted):
        ax.text(val + max(values_sorted) * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f} W", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Mean Absolute Error (Watts)", fontsize=12)
    ax.set_title("Model Comparison: MAE on Drone Power Consumption (Test Set)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, max(values_sorted) * 1.18)
    ax.invert_yaxis()

    # Chú thích mô hình tốt nhất
    ax.axvline(values_sorted[0], color="#F44336", linestyle="--", alpha=0.5, lw=1.5)
    ax.legend(handles=[mpatches.Patch(color="#F44336", label=f"Best: {names_sorted[0]}")],
              loc="lower right", fontsize=10)

    fig.tight_layout()
    _save(fig, "01_mae_comparison.png")


def plot_02_xai_fidelity(fidelity_scores: dict) -> None:
    """Bar chart so sánh điểm Kendall-Tau (XAI Fidelity)."""
    names  = list(fidelity_scores.keys())
    values = [fidelity_scores[n] for n in names]
    order  = np.argsort(values)[::-1]    # sắp xếp giảm dần

    names_sorted  = [names[i]  for i in order]
    values_sorted = [values[i] for i in order]
    colors        = [PALETTE_5[i % len(PALETTE_5)] for i in range(len(names_sorted))]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names_sorted, values_sorted, color=colors, edgecolor="white",
                  linewidth=0.8, width=0.55)

    # Nhãn giá trị
    for bar, val in zip(bars, values_sorted):
        ypos = val + 0.01 if val >= 0 else val - 0.03
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-0.1, 1.15)
    ax.set_ylabel("Kendall-Tau Correlation Coefficient", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("XAI Fidelity: Kendall-Tau vs. Ridge Ground Truth\n"
                 "(Higher = More Faithful Feature Ranking)", fontsize=14,
                 fontweight="bold", pad=15)

    # Vùng tham chiếu: τ > 0.7 = high fidelity
    ax.axhspan(0.7, 1.15, alpha=0.08, color="green", label="High Fidelity (τ > 0.7)")
    ax.legend(fontsize=10)

    fig.tight_layout()
    _save(fig, "02_xai_fidelity.png")


def plot_03_shap_summary(
    best_model_name: str,
    fitted_models: dict,
    X_test: np.ndarray,
    feature_names: list,
    shap_values_dict: dict,
) -> None:
    """SHAP summary plot (beeswarm) cho mô hình có MAE tốt nhất."""
    import shap

    sv = shap_values_dict[best_model_name]

    # Tạo DataFrame để shap biết tên cột
    # Lưu ý: KernelExplainer có thể đã dùng X_explain (subset), cần đồng bộ shape
    n_samples = sv.shape[0]
    X_display = X_test[:n_samples]

    import pandas as _pd
    X_df = _pd.DataFrame(X_display, columns=feature_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        sv, X_df,
        feature_names = feature_names,
        plot_type     = "dot",     # beeswarm
        show          = False,
        max_display   = len(feature_names),
    )
    plt.title(f"SHAP Summary Plot – {best_model_name} (Best MAE Model)\n"
              f"Feature Impact on Drone Power Consumption",
              fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("SHAP Value (impact on model output)", fontsize=12)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "03_shap_summary_best_model.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Lưu: {path}")


def plot_04_importance_vs_groundtruth(
    ground_truth: dict,
    shap_importances: dict,
    best_model_name: str,
    top_n: int = 5,
) -> None:
    """
    Grouped Bar Chart: so sánh % tầm quan trọng của top-N feature
    giữa Ridge Ground Truth và mô hình best (SHAP).
    """
    # Chọn top_n feature theo Ground Truth
    gt_sorted = sorted(ground_truth.items(), key=lambda x: -x[1])
    top_features = [f for f, _ in gt_sorted[:top_n]]

    best_imp = shap_importances[best_model_name]

    gt_vals   = [ground_truth[f]  for f in top_features]
    shap_vals = [best_imp.get(f, 0.0) for f in top_features]

    x     = np.arange(len(top_features))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars1 = ax.bar(x - width/2, gt_vals,   width, label="Ridge (Ground Truth)",
                   color="#2196F3", edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width/2, shap_vals, width, label=f"{best_model_name} (SHAP)",
                   color="#F44336", edgecolor="white", linewidth=0.8)

    # Nhãn giá trị
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(top_features, fontsize=11)
    ax.set_ylabel("Feature Importance (%)", fontsize=12)
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_title(
        f"Feature Importance: Ridge Ground Truth vs {best_model_name} (SHAP)\n"
        f"Top {top_n} Features by Ridge Coefficient Magnitude",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim(0, max(gt_vals + shap_vals) * 1.22)

    fig.tight_layout()
    _save(fig, "04_feature_importance_vs_groundtruth.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  DRONE POWER CONSUMPTION – ML + XAI FIDELITY PIPELINE")
    print("█"*60)

    # 0. Tạo thư mục output
    setup_results_dir(RESULTS_DIR)

    # 1. Nạp & xử lý dữ liệu
    X_train, X_test, y_train, y_test, scaler, feature_names = step1_load_data()

    # 2. Huấn luyện & đánh giá MAE
    fitted_models, mae_scores = step2_train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    # Xác định mô hình tốt nhất (MAE thấp nhất)
    best_model_name = min(mae_scores, key=mae_scores.get)
    print(f"\n  ★ Mô hình tốt nhất theo MAE: {best_model_name} "
          f"(MAE = {mae_scores[best_model_name]:.4f} W)")

    # 3. XAI: Ground Truth + SHAP + Fidelity
    ground_truth, shap_importances, shap_values_dict, fidelity_scores = step3_xai(
        fitted_models, X_train, X_test, feature_names
    )

    # 4. Lưu bảng CSV
    df_mae, df_fid = step4_save_tables(mae_scores, fidelity_scores)

    # 5. Trực quan hóa
    print("\n" + "="*60)
    print("BƯỚC 5: Sinh hình ảnh (4 biểu đồ)")
    print("="*60)

    print("\n  [5.1] Bar chart MAE ...")
    plot_01_mae_comparison(mae_scores)

    print("  [5.2] Bar chart XAI Fidelity (Kendall-Tau) ...")
    plot_02_xai_fidelity(fidelity_scores)

    print("  [5.3] SHAP Summary Plot (best model) ...")
    plot_03_shap_summary(
        best_model_name, fitted_models, X_test, feature_names, shap_values_dict
    )

    print("  [5.4] Feature Importance Grouped Bar Chart ...")
    plot_04_importance_vs_groundtruth(
        ground_truth, shap_importances, best_model_name, top_n=5
    )

    # 6. Đánh giá tính bền bỉ (Robustness)
    print("\n" + "="*60)
    print("BƯỚC 6: Đánh giá Robustness (Bơm nhiễu Gaussian)")
    print("="*60)
    from robustness_evaluator import evaluate_robustness, plot_and_save_robustness
    
    # Chạy mô phỏng nhiễu
    df_robustness = evaluate_robustness(fitted_models, X_test, y_test)
    
    # Lưu file CSV và vẽ biểu đồ số 5
    plot_and_save_robustness(df_robustness, RESULTS_DIR)

    # ── Tóm tắt cuối ─────────────────────────────────────────────────────────
    print("\n" + "█"*60)
    print("  PIPELINE HOÀN THÀNH – KẾT QUẢ TỔNG KẾT")
    print("█"*60)
    print("\n  MAE (sắp xếp từ tốt đến kém):")
    print(df_mae.to_string(index=False))
    print("\n  Fidelity / Kendall-Tau (sắp xếp từ cao đến thấp):")
    print(df_fid.to_string(index=False))
    print(f"\n  Tất cả output đã được lưu vào: '{RESULTS_DIR}/'")
    print("█"*60 + "\n")

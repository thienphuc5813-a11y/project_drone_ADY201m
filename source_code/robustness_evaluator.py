"""
robustness_evaluator.py
========================
Đánh giá tính bền bỉ (Robustness) của các mô hình hộp đen (Ridge, MLP,
RandomForest, GradientBoosting, CatBoost) bằng cách bơm nhiễu Gaussian vào
tập kiểm tra (X_test) và đo lường sự suy giảm hiệu năng dựa trên MAE.

Pipeline:
    1. add_gaussian_noise        -> bơm nhiễu Gaussian theo từng mức độ.
    2. evaluate_robustness       -> đánh giá MAE của 5 mô hình tại mỗi mức nhiễu.
    3. plot_and_save_robustness  -> lưu CSV tổng hợp + vẽ biểu đồ đường (academic style).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # Bắt buộc: tránh lỗi backend GUI khi chạy trên server

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error

# --------------------------------------------------------------------------- #
# Cấu hình logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("robustness_evaluator")


# --------------------------------------------------------------------------- #
# 1. BƠM NHIỄU GAUSSIAN
# --------------------------------------------------------------------------- #
def add_gaussian_noise(
    X: np.ndarray,
    noise_level: float,
    random_state: int = 42,
) -> np.ndarray:
    """
    Bơm nhiễu Gaussian vào dữ liệu theo công thức:
        X_noisy = X + noise_level * std(X) * N(0, 1)

    Trong đó std(X) là độ lệch chuẩn tính theo từng cột (feature) của X,
    và N(0, 1) là nhiễu lấy từ phân phối chuẩn tắc, cùng shape với X.

    Args:
        X: Ma trận dữ liệu gốc, shape (n_samples, n_features).
        noise_level: Hệ số cường độ nhiễu (0.0 = không nhiễu, 0.2 = 20% std).
        random_state: Seed để đảm bảo kết quả có thể tái lập.

    Returns:
        np.ndarray: Ma trận dữ liệu đã bị bơm nhiễu, cùng shape với X.
    """
    X_arr = np.asarray(X, dtype=np.float64)

    if noise_level == 0.0:
        logger.info("noise_level = 0.0 -> trả về dữ liệu gốc, không bơm nhiễu.")
        return X_arr.copy()

    rng = np.random.RandomState(random_state)
    col_std = X_arr.std(axis=0)  # độ lệch chuẩn theo từng cột
    noise = rng.normal(loc=0.0, scale=1.0, size=X_arr.shape)

    X_noisy = X_arr + noise_level * col_std * noise

    logger.info(
        "Đã bơm nhiễu Gaussian với noise_level=%.2f vào dữ liệu shape=%s.",
        noise_level,
        X_arr.shape,
    )
    return X_noisy


# --------------------------------------------------------------------------- #
# 2. ĐÁNH GIÁ ROBUSTNESS CỦA 5 MÔ HÌNH
# --------------------------------------------------------------------------- #
def evaluate_robustness(
    fitted_models: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_levels: List[float] = [0.0, 0.05, 0.10, 0.20],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Đánh giá độ bền bỉ của nhiều mô hình hộp đen bằng cách đo MAE trên
    dữ liệu kiểm tra tại các mức độ nhiễu Gaussian khác nhau.

    Args:
        fitted_models: Dict {tên_mô_hình: mô_hình_đã_huấn_luyện (.predict())}.
        X_test: Ma trận đặc trưng kiểm tra, shape (n_samples, n_features).
        y_test: Vector mục tiêu thực tế, shape (n_samples,).
        noise_levels: Danh sách các mức độ nhiễu cần thử (0.0 = dữ liệu gốc).
        random_state: Seed để đảm bảo kết quả nhiễu có thể tái lập.

    Returns:
        pd.DataFrame: Cột ['Model', 'Noise_Level', 'MAE'], mỗi dòng là một
        cặp (mô hình, mức nhiễu) cùng giá trị MAE tương ứng.
    """
    logger.info("=" * 70)
    logger.info("BẮT ĐẦU ĐÁNH GIÁ ROBUSTNESS CHO %d MÔ HÌNH", len(fitted_models))
    logger.info("Các mức độ nhiễu sẽ kiểm tra: %s", noise_levels)
    logger.info("=" * 70)

    records: List[Dict[str, object]] = []

    for noise_level in noise_levels:
        logger.info("-" * 70)
        logger.info("Mức nhiễu: %.2f", noise_level)
        logger.info("-" * 70)

        X_noisy = add_gaussian_noise(X_test, noise_level, random_state=random_state)

        for model_name, model in fitted_models.items():
            y_pred = model.predict(X_noisy)
            mae = float(mean_absolute_error(y_test, y_pred))

            records.append(
                {
                    "Model": model_name,
                    "Noise_Level": noise_level,
                    "MAE": mae,
                }
            )

            logger.info("   - %-20s : MAE = %.4f", model_name, mae)

    df_robustness = pd.DataFrame(records, columns=["Model", "Noise_Level", "MAE"])

    logger.info("=" * 70)
    logger.info("HOÀN TẤT ĐÁNH GIÁ ROBUSTNESS - tổng %d dòng kết quả.", len(df_robustness))
    logger.info("=" * 70)

    return df_robustness


# --------------------------------------------------------------------------- #
# 3. LƯU CSV + VẼ BIỂU ĐỒ ROBUSTNESS
# --------------------------------------------------------------------------- #
def plot_and_save_robustness(
    df_robustness: pd.DataFrame,
    results_dir: str,
) -> None:
    """
    Lưu DataFrame robustness thành CSV và vẽ biểu đồ đường thể hiện sự suy
    giảm MAE của từng mô hình theo mức độ nhiễu Gaussian (academic style).

    Args:
        df_robustness: DataFrame với các cột ['Model', 'Noise_Level', 'MAE'].
        results_dir: Đường dẫn thư mục để lưu file CSV và biểu đồ PNG.

    Returns:
        None.
    """
    logger.info("=" * 70)
    logger.info("LƯU KẾT QUẢ ROBUSTNESS VÀO THƯ MỤC: %s", results_dir)
    logger.info("=" * 70)

    os.makedirs(results_dir, exist_ok=True)

    # ---- 1. Lưu CSV ----
    csv_path = os.path.join(results_dir, "summary_robustness.csv")
    df_robustness.to_csv(csv_path, index=False)
    logger.info("Đã lưu CSV tổng hợp tại: %s", csv_path)

    # ---- 2. Thiết lập style học thuật ----
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.lineplot(
        data=df_robustness,
        x="Noise_Level",
        y="MAE",
        hue="Model",
        marker="o",
        markersize=8,
        linewidth=2,
        ax=ax,
    )

    ax.set_title("Model Robustness: MAE Degradation under Gaussian Noise", fontweight="bold")
    ax.set_xlabel("Noise Level")
    ax.set_ylabel("Mean Absolute Error (MAE)")

    # Trục X hiển thị dạng phần trăm (0%, 5%, 10%, 20%...)
    noise_ticks = sorted(df_robustness["Noise_Level"].unique())
    ax.set_xticks(noise_ticks)
    ax.set_xticklabels([f"{int(round(n * 100))}%" for n in noise_ticks])

    # Bỏ viền trên/phải, grid mờ
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)

    ax.legend(title="Model", frameon=False, loc="best")

    fig.tight_layout()

    # ---- 3. Lưu biểu đồ ----
    plot_path = os.path.join(results_dir, "05_robustness_degradation.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)  # Chống tràn RAM đồ họa

    logger.info("Đã lưu biểu đồ robustness tại: %s", plot_path)
    logger.info("Hoàn tất plot_and_save_robustness().")

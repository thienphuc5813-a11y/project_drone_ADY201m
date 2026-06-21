"""
xai_evaluator.py
=================
Hệ thống đánh giá độ tin cậy (Fidelity) của các phương pháp giải thích mô hình
(XAI) bằng cách so sánh Feature Importance trích xuất từ SHAP với "Đáp án
chuẩn vật lý" (Ground Truth) được xây dựng từ hệ số hồi quy Ridge.

Tối ưu cho tập dữ liệu lớn (>140,000 dòng) để chống tràn RAM (OOM):
    - Downcasting dữ liệu về float32 (giảm ~50% RAM).
    - Lấy mẫu (sampling) dữ liệu trước khi đưa vào SHAP Explainer.
    - Dùng shap.kmeans để nén background data cho KernelExplainer.
    - Giải phóng RAM chủ động bằng gc.collect() sau mỗi mô hình.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import shap
from scipy.stats import kendalltau
from sklearn.linear_model import Ridge

# --------------------------------------------------------------------------- #
# Cấu hình logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("xai_evaluator")

# Phân nhóm mô hình để chọn đúng loại SHAP Explainer
TREE_BASED_MODELS = {"randomforest", "gradientboosting", "catboost"}
SMOOTH_MODELS = {"ridge", "mlpregressor"}


# --------------------------------------------------------------------------- #
# 1. GROUND TRUTH
# --------------------------------------------------------------------------- #
def build_soft_ground_truth(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Xây dựng "Đáp án chuẩn vật lý" (Ground Truth) dựa trên hệ số hồi quy Ridge.

    Quy trình:
        1. Huấn luyện Ridge(alpha=1.0) trên (X_train, y_train).
        2. Lấy trị tuyệt đối của coef_.
        3. Chuẩn hóa về tỷ lệ phần trăm (tổng = 100%).

    Args:
        X_train: Ma trận đặc trưng huấn luyện, shape (n_samples, n_features).
        y_train: Vector mục tiêu, shape (n_samples,).
        feature_names: Danh sách tên các biến, độ dài = n_features.

    Returns:
        Dict[str, float]: {tên_biến: phần trăm quan trọng}.
    """
    logger.info("=" * 70)
    logger.info("BƯỚC 1: Xây dựng Ground Truth bằng hệ số hồi quy Ridge(alpha=1.0)")
    logger.info("=" * 70)

    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X_train, y_train)
    logger.info(
        "Đã huấn luyện xong Ridge trên %d mẫu, %d biến.",
        X_train.shape[0],
        X_train.shape[1],
    )

    abs_coef = np.abs(np.asarray(ridge_model.coef_, dtype=np.float64)).ravel()
    total = float(abs_coef.sum())

    if total == 0.0:
        logger.warning(
            "Tổng |coef_| = 0 -> chia đều %% cho tất cả biến để tránh chia cho 0."
        )
        n_features = len(feature_names)
        percentages = (
            np.full(n_features, 100.0 / n_features) if n_features > 0 else np.array([])
        )
    else:
        percentages = (abs_coef / total) * 100.0

    ground_truth: Dict[str, float] = {
        name: float(pct) for name, pct in zip(feature_names, percentages)
    }

    top5 = sorted(ground_truth.items(), key=lambda kv: kv[1], reverse=True)[:5]
    logger.info("Ground Truth - Top 5 biến quan trọng nhất:")
    for name, pct in top5:
        logger.info("   - %-25s : %6.2f%%", name, pct)

    return ground_truth


# --------------------------------------------------------------------------- #
# 2. SHAP IMPORTANCE (tối ưu RAM)
# --------------------------------------------------------------------------- #
def extract_shap_importance(
    model_name: str,
    fitted_model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Trích xuất Feature Importance bằng SHAP cho một mô hình đã huấn luyện,
    được tối ưu chống tràn RAM cho tập dữ liệu lớn (>140,000 dòng).

    Tối ưu RAM áp dụng:
        - Downcast X_train, X_test sang float32.
        - Chỉ lấy tối đa 500 dòng ngẫu nhiên từ X_test để giải thích.
        - Với mô hình "trơn" (Ridge, MLPRegressor): nén background bằng
          shap.kmeans(..., 50) từ 5000 dòng train được lấy mẫu ngẫu nhiên,
          thay vì dùng toàn bộ dữ liệu train.
        - Giải phóng RAM ngay sau khi tính xong bằng gc.collect().

    Args:
        model_name: Tên mô hình (ví dụ: "RandomForest", "Ridge", "CatBoost"...).
        fitted_model: Đối tượng mô hình đã được .fit().
        X_train: Dữ liệu huấn luyện (dùng làm background cho KernelExplainer).
        X_test: Dữ liệu kiểm tra (dùng để tính SHAP values).
        feature_names: Danh sách tên biến, độ dài = n_features.
        random_state: Seed để đảm bảo kết quả lấy mẫu có thể tái lập.

    Returns:
        Dict[str, float]: {tên_biến: phần trăm quan trọng theo SHAP}.
    """
    logger.info("-" * 70)
    logger.info("Đang trích xuất SHAP importance cho mô hình: %s", model_name)
    logger.info("-" * 70)

    rng = np.random.RandomState(random_state)
    model_key = model_name.strip().lower()

    explainer: Optional[Any] = None
    background: Optional[Any] = None

    # ---- Downcasting để giảm ~50% RAM ----
    X_train_f32 = np.asarray(X_train, dtype=np.float32)
    X_test_f32 = np.asarray(X_test, dtype=np.float32)
    logger.info("Đã downcast X_train, X_test về float32.")

    # ---- Sampling tối đa 500 dòng từ X_test ----
    n_test = X_test_f32.shape[0]
    test_sample_size = min(500, n_test)
    test_idx = rng.choice(n_test, size=test_sample_size, replace=False)
    X_test_sample = X_test_f32[test_idx]
    logger.info(
        "Đã lấy mẫu %d/%d dòng từ X_test để đưa vào SHAP Explainer.",
        test_sample_size,
        n_test,
    )

    try:
        if model_key in TREE_BASED_MODELS:
            logger.info("Nhóm CÂY phát hiện -> dùng shap.TreeExplainer.")
            explainer = shap.TreeExplainer(fitted_model)
            raw_shap = explainer.shap_values(X_test_sample)

        else:
            # Nhóm TRƠN (Ridge, MLPRegressor) hoặc mô hình lạ -> bắt buộc KernelExplainer
            if model_key in SMOOTH_MODELS:
                logger.info("Nhóm TRƠN phát hiện -> dùng shap.KernelExplainer.")
            else:
                logger.warning(
                    "Mô hình '%s' không thuộc nhóm CÂY/TRƠN đã biết -> "
                    "fallback an toàn dùng shap.KernelExplainer.",
                    model_name,
                )

            n_train = X_train_f32.shape[0]
            bg_sample_size = min(5000, n_train)
            bg_idx = rng.choice(n_train, size=bg_sample_size, replace=False)
            X_train_sampled = X_train_f32[bg_idx]
            logger.info(
                "Đã lấy mẫu %d/%d dòng từ X_train làm cơ sở cho background.",
                bg_sample_size,
                n_train,
            )

            n_clusters = min(50, bg_sample_size)
            background = shap.kmeans(X_train_sampled, n_clusters)
            logger.info("Đã nén background data thành %d cụm (k-means).", n_clusters)

            predict_fn = getattr(fitted_model, "predict", None)
            if predict_fn is None:
                raise AttributeError(
                    f"Mô hình '{model_name}' không có phương thức predict()."
                )

            explainer = shap.KernelExplainer(predict_fn, background)
            raw_shap = explainer.shap_values(X_test_sample, silent=True)

        # ---- Chuẩn hóa output (một số mô hình trả về list cho multi-output) ----
        if isinstance(raw_shap, list):
            shap_values = np.abs(np.asarray(raw_shap[0], dtype=np.float64))
        else:
            shap_values = np.abs(np.asarray(raw_shap, dtype=np.float64))

        # ---- Tính mean(|shap|) theo từng biến và chuẩn hóa % ----
        mean_abs_shap = shap_values.mean(axis=0).ravel()
        total = float(mean_abs_shap.sum())

        if total == 0.0:
            logger.warning(
                "Tổng mean(|SHAP|) = 0 cho mô hình '%s' -> chia đều %% để tránh "
                "lỗi chia cho 0.",
                model_name,
            )
            n_features = len(feature_names)
            percentages = (
                np.full(n_features, 100.0 / n_features)
                if n_features > 0
                else np.array([])
            )
        else:
            percentages = (mean_abs_shap / total) * 100.0

        importance_dict: Dict[str, float] = {
            name: float(pct) for name, pct in zip(feature_names, percentages)
        }

        top3 = sorted(importance_dict.items(), key=lambda kv: kv[1], reverse=True)[:3]
        logger.info("SHAP importance - Top 3 biến cho '%s':", model_name)
        for name, pct in top3:
            logger.info("   - %-25s : %6.2f%%", name, pct)

        return importance_dict, raw_shap

    finally:
        # ---- Dọn rác để ép hệ điều hành thu hồi RAM ngay lập tức ----
        if explainer is not None:
            del explainer
        if background is not None:
            del background
        del X_train_f32, X_test_f32, X_test_sample
        gc.collect()
        logger.info("Đã giải phóng RAM (gc.collect()) sau khi xử lý '%s'.", model_name)


# --------------------------------------------------------------------------- #
# 3. FIDELITY SCORING
# --------------------------------------------------------------------------- #
def compute_fidelity_scores(
    shap_importances_dict: Dict[str, Dict[str, float]],
    ground_truth_dict: Dict[str, float],
) -> Dict[str, float]:
    """
    Chấm điểm "XAI Fidelity" cho từng mô hình bằng hệ số tương quan Kendall-Tau
    giữa thứ hạng Feature Importance của SHAP và thứ hạng của Ground Truth (Ridge).

    Args:
        shap_importances_dict: {tên_mô_hình: {tên_biến: % quan trọng theo SHAP}}.
        ground_truth_dict: {tên_biến: % quan trọng theo Ground Truth}.

    Returns:
        Dict[str, float]: {tên_mô_hình: hệ số Kendall-Tau}.
    """
    logger.info("=" * 70)
    logger.info("BƯỚC 3: Tính điểm XAI Fidelity (Kendall-Tau) cho từng mô hình")
    logger.info("=" * 70)

    gt_features = list(ground_truth_dict.keys())
    fidelity_scores: Dict[str, float] = {}

    for model_name, shap_importance in shap_importances_dict.items():
        common_features = [f for f in gt_features if f in shap_importance]

        if len(common_features) < 2:
            logger.warning(
                "Mô hình '%s' không đủ biến chung (%d) với Ground Truth để tính "
                "Kendall-Tau -> gán NaN.",
                model_name,
                len(common_features),
            )
            fidelity_scores[model_name] = float("nan")
            continue

        shap_arr = np.array([shap_importance[f] for f in common_features])
        gt_arr = np.array([ground_truth_dict[f] for f in common_features])

        if np.all(shap_arr == shap_arr[0]) or np.all(gt_arr == gt_arr[0]):
            logger.warning(
                "Mô hình '%s' có mảng giá trị không đổi (constant) -> Kendall-Tau "
                "không xác định, gán 0.0 để tránh lỗi.",
                model_name,
            )
            fidelity_scores[model_name] = 0.0
            continue

        tau, p_value = kendalltau(shap_arr, gt_arr)
        tau = 0.0 if tau is None or np.isnan(tau) else float(tau)
        fidelity_scores[model_name] = tau

        logger.info(
            "   - %-20s : Kendall-Tau = %6.4f (p-value = %.4f)",
            model_name,
            tau,
            float(p_value) if p_value is not None and not np.isnan(p_value) else float("nan"),
        )

    logger.info("Hoàn tất chấm điểm XAI Fidelity cho %d mô hình.", len(fidelity_scores))
    return fidelity_scores

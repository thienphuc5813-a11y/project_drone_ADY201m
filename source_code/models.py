"""
models.py
=========
Định nghĩa 5 "Đấu sĩ" (mô hình chưa fit) trả về dưới dạng dict.

Nhóm Trơn (Linear/Neural):
  - Ridge           : baseline tuyến tính, coef_ có tính vật lý rõ ràng
  - MLPRegressor    : mạng nơ-ron 2 lớp ẩn (64 → 32)

Nhóm Cây (Tree-based):
  - RandomForestRegressor
  - GradientBoostingRegressor
  - CatBoostRegressor
"""

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# CatBoost – thử import, báo lỗi rõ ràng nếu chưa cài
try:
    from catboost import CatBoostRegressor
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False
    print("[models] Cảnh báo: CatBoost chưa được cài. "
          "Chạy `pip install catboost` để dùng mô hình này.")


def get_models(random_state: int = 42) -> dict:
    """
    Trả về dict { tên_model: instance_model_chưa_fit }.

    Parameters
    ----------
    random_state : int  – seed ngẫu nhiên chung cho tái lập kết quả

    Returns
    -------
    dict[str, estimator]
    """
    models = {}

    # ── Nhóm Trơn ────────────────────────────────────────────────────────────

    # Ridge: alpha=1.0 (L2 regularization nhẹ), dùng làm baseline vật lý
    models["Ridge"] = Ridge(alpha=1.0)

    # MLP: 2 lớp ẩn [64 → 32], relu, adam, early_stopping để tránh overfit
    models["MLPRegressor"] = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
        batch_size=512,
    )

    # ── Nhóm Cây ─────────────────────────────────────────────────────────────

    # Random Forest: 300 cây, giới hạn độ sâu để kiểm soát phức tạp
    models["RandomForest"] = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,         # cho phép cây phát triển đầy đủ
        min_samples_leaf=5,
        n_jobs=-1,              # dùng tất cả CPU cores
        random_state=random_state,
    )

    # Gradient Boosting: 300 vòng, learning_rate thấp + subsample tránh overfit
    models["GradientBoosting"] = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=random_state,
    )

    # CatBoost: tự xử lý tốt dữ liệu, silent=True để tắt log dài dòng
    if _CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            eval_metric="MAE",
            random_seed=random_state,
            verbose=0,          # tắt output training
        )
    else:
        # Fallback: dùng thêm 1 GradientBoosting với tham số khác
        models["CatBoost"] = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.7,
            random_state=random_state,
        )
        print("[models] Fallback: CatBoost thay bằng GradientBoosting (tham số khác).")

    print(f"[models] Đã tạo {len(models)} mô hình: {list(models.keys())}")
    return models

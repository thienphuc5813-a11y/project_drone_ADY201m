"""
dataset.py
==========
Xử lý dữ liệu: nạp CSV, One-Hot Encoding, chia Train/Test theo GroupShuffleSplit
để chống data leakage theo cột `flight`, và chuẩn hóa features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


# ── Cấu hình các cột ─────────────────────────────────────────────────────────
FEATURE_COLS = [
    "payload", "speed", "altitude", "wind_speed",
    "velocity_mag", "accel_mag", "angular_mag", "pitch", "headwind"
]
TARGET_COL   = "Power"
GROUP_COL    = "flight"      # dùng để chống leakage khi chia train/test
REGIME_COL   = "regime"      # sẽ được One-Hot Encoding
# ─────────────────────────────────────────────────────────────────────────────


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """
    Nạp CSV và thực hiện One-Hot Encoding cho cột `regime`.

    Parameters
    ----------
    csv_path : str
        Đường dẫn tới file CSV.

    Returns
    -------
    df : pd.DataFrame
        DataFrame đã được xử lý, bao gồm cột `flight` và `Power`.
    """
    df = pd.read_csv(csv_path)

    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    print(f"[dataset] Đã nạp dữ liệu: {df.shape[0]:,} hàng × {df.shape[1]} cột")

    # ── One-Hot Encoding cột `regime` ────────────────────────────────────────
    if REGIME_COL in df.columns:
        dummies = pd.get_dummies(df[REGIME_COL], prefix=REGIME_COL, drop_first=False)
        # Ép kiểu sang int (0/1) cho sạch
        dummies = dummies.astype(int)
        df = pd.concat([df.drop(columns=[REGIME_COL]), dummies], axis=1)
        print(f"[dataset] One-Hot Encoding `{REGIME_COL}` → {dummies.shape[1]} cột mới: "
              f"{list(dummies.columns)}")
    else:
        print(f"[dataset] Cảnh báo: không tìm thấy cột `{REGIME_COL}`.")

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Trả về danh sách TẤT CẢ feature columns = FEATURE_COLS + các cột OHE từ regime.

    Parameters
    ----------
    df : pd.DataFrame  (đã qua load_and_preprocess)

    Returns
    -------
    list of str
    """
    regime_ohe_cols = [c for c in df.columns if c.startswith(f"{REGIME_COL}_")]
    all_features = FEATURE_COLS + regime_ohe_cols
    # Chỉ giữ các cột thực sự tồn tại trong df
    all_features = [c for c in all_features if c in df.columns]
    return all_features


def split_and_scale(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Chia train/test theo GroupShuffleSplit (nhóm theo `flight` để chống leakage),
    rồi áp dụng StandardScaler trên X.

    Parameters
    ----------
    df           : pd.DataFrame  đã qua load_and_preprocess
    test_size    : float         tỉ lệ tập test (mặc định 20 %)
    random_state : int

    Returns
    -------
    X_train_sc, X_test_sc : np.ndarray  – features đã scale
    y_train, y_test       : np.ndarray  – target (Power)
    scaler                : StandardScaler đã fit
    feature_names         : list[str]   – tên features theo thứ tự cột
    """
    feature_names = get_feature_columns(df)

    X     = df[feature_names].values
    y     = df[TARGET_COL].values
    groups = df[GROUP_COL].values           # nhóm theo flight_id

    # ── GroupShuffleSplit: đảm bảo cùng 1 chuyến bay KHÔNG chia 2 tập ───────
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Kiểm tra: không có flight nào vừa ở train vừa ở test
    train_flights = set(groups[train_idx])
    test_flights  = set(groups[test_idx])
    overlap = train_flights & test_flights
    assert len(overlap) == 0, f"Phát hiện {len(overlap)} flight bị overlap! {overlap}"

    print(f"[dataset] Train: {len(train_idx):,} mẫu | {len(train_flights)} flights")
    print(f"[dataset] Test : {len(test_idx):,} mẫu | {len(test_flights)} flights")
    print(f"[dataset] Overlap flights = {len(overlap)} ✓ (không có leakage)")

    # ── StandardScaler: fit trên train, transform cả 2 ───────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"[dataset] Features ({len(feature_names)}): {feature_names}")
    return X_train_sc, X_test_sc, y_train, y_test, scaler, feature_names

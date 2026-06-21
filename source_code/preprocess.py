"""
prepare_data_eda.py

Mục đích:
    Đọc dữ liệu thô từ real_data/flights_processed.csv, thực hiện Feature
    Engineering (tạo Target Power, các biến Magnitude, góc pitch, headwind),
    sau đó CHỈ GIỮ LẠI các cột không gây rò rỉ dữ liệu (Data Leakage) để
    phục vụ bước Exploratory Data Analysis (EDA).

    Lưu ý quan trọng: Script này KHÔNG chia Train/Test, KHÔNG Encode,
    KHÔNG Scale. Toàn bộ các bước đó sẽ được thực hiện ở giai đoạn sau,
    sau khi đã quan sát dữ liệu qua EDA.
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def build_eda_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Đọc file CSV gốc, lọc nhiễu cơ bản và trả về DataFrame 
    đã qua Feature Engineering, sẵn sàng cho bước EDA.

    Tham số:
        csv_path (str): Đường dẫn tới file flights_processed.csv gốc.

    Trả về:
        pd.DataFrame: DataFrame chỉ gồm các cột an toàn (không rò rỉ),
        đã có đầy đủ các đặc trưng kỹ thuật (engineered features).
    """

    # ----------------------------------------------------------------
    # BƯỚC 1: Đọc dữ liệu thô và Lọc nhiễu
    # ----------------------------------------------------------------
    df = pd.read_csv(csv_path, low_memory=False)
    
    # [THÊM MỚI] Lọc bỏ chuyến bay biến thiên độ cao (nếu vô tình lọt vào)
    df = df[df["altitude"].astype(str) != "25-50-100-25"].copy()
    
    # [THÊM MỚI] Lọc bỏ các chuyến bay bị lỗi (quá ngắn, dưới 20 dòng)
    df = df.groupby('flight').filter(lambda x: len(x) >= 20).copy()

    # ----------------------------------------------------------------
    # BƯỚC 2: Tạo biến mục tiêu (Target) - Power
    # ----------------------------------------------------------------
    df["Power"] = df["battery_voltage"] * df["battery_current"]

    # ----------------------------------------------------------------
    # BƯỚC 3: Tính các biến Magnitude (độ lớn tổng hợp 3 trục x, y, z)
    # ----------------------------------------------------------------

    # 3.1. Độ lớn vận tốc (velocity_mag)
    df["velocity_mag"] = np.sqrt(
        df["velocity_x"] ** 2 + df["velocity_y"] ** 2 + df["velocity_z"] ** 2
    )

    # 3.2. Độ lớn gia tốc (accel_mag) 
    df["accel_mag"] = np.sqrt(
        df["linear_acceleration_x"] ** 2
        + df["linear_acceleration_y"] ** 2
        + df["linear_acceleration_z"] ** 2
    )

    # 3.3. Độ lớn vận tốc góc (angular_mag) 
    df["angular_mag"] = np.sqrt(
        df["angular_x"] ** 2 + df["angular_y"] ** 2 + df["angular_z"] ** 2
    )

    # ----------------------------------------------------------------
    # BƯỚC 4: Dịch Quaternion (orientation) sang góc Euler (Pitch)
    # ----------------------------------------------------------------
    quaternions = df[
        ["orientation_x", "orientation_y", "orientation_z", "orientation_w"]
    ].to_numpy()

    euler_angles = R.from_quat(quaternions).as_euler("xyz", degrees=True)
    df["pitch"] = euler_angles[:, 1]  

    # ----------------------------------------------------------------
    # BƯỚC 5: Tính lực cản gió ngược (headwind)
    # ----------------------------------------------------------------
    df["headwind"] = df["wind_speed"] * np.cos(np.radians(df["wind_angle"]))

    # ----------------------------------------------------------------
    # BƯỚC 6: Lựa chọn đặc trưng cuối cùng (Chống rò rỉ dữ liệu)
    # ----------------------------------------------------------------
    final_columns = [
        "flight",
        "payload",
        "speed",
        "altitude",
        "wind_speed",
        "regime",
        "velocity_mag",
        "accel_mag",
        "angular_mag",
        "pitch",
        "headwind",
        "Power",
    ]

    # Loại bỏ các dòng bị NaN (nếu có) phát sinh trong quá trình tính toán
    df_eda = df[final_columns].dropna().copy()

    return df_eda


if __name__ == "__main__":
    input_path = "real_data/flights_processed.csv"
    output_path = "real_data/flights_ready_for_eda.csv"

    eda_df = build_eda_dataframe(input_path)

    eda_df.to_csv(output_path, index=False)

    print(f"Đã lưu file vào: {output_path}")
    print(f"Kích thước DataFrame: {eda_df.shape}")
    print("\n5 dòng đầu tiên:")
    print(eda_df.head())
import streamlit as st
import joblib
import numpy as np
import os

# -----------------------------
# Thiết lập cấu hình trang
# -----------------------------
st.set_page_config(
    page_title="Dự đoán học lực học sinh",
    page_icon="📚",
    layout="centered"
)

# -----------------------------
# CSS tùy chỉnh giao diện
# -----------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            padding: 0.5em 1em;
        }
        .stNumberInput>div>div>input {
            background-color: #ffffff !important;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
#  Tải mô hình và scaler
# -----------------------------
if os.path.exists("student_model.pkl") and os.path.exists("student_scaler.pkl"):
    model = joblib.load("student_model.pkl")
    scaler = joblib.load("student_scaler.pkl")
else:
    st.error("❌ Không tìm thấy file mô hình hoặc scaler. Vui lòng kiểm tra lại.")
    st.stop()

# -----------------------------
# Tiêu đề ứng dụng
# -----------------------------
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>📚 Ứng dụng dự đoán học lực học sinh</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Dựa vào thói quen học tập hàng ngày</p>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# Biểu mẫu nhập thông tin
# -----------------------------
with st.form("student_form"):
    st.subheader("📝 Nhập thông tin học sinh:")

    col1, col2 = st.columns(2)
    with col1:
        hours = st.number_input("📖 Số giờ học mỗi ngày", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
        sleep = st.number_input("😴 Số giờ ngủ mỗi ngày", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
        activity = st.selectbox("🏀 Tham gia hoạt động ngoại khóa?", ["Không", "Có"])

    with col2:
        previous = st.number_input("📊 Điểm trung bình năm trước", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        papers = st.number_input("📝 Số đề luyện tập đã làm", min_value=0, max_value=50, value=5, step=1)

    submit_button = st.form_submit_button("📊 Dự đoán học lực")

# -----------------------------
# Xử lý khi nhấn nút
# -----------------------------
if submit_button:
    activity_num = 1 if activity == "Có" else 0
    input_data = np.array([[hours, previous, activity_num, sleep, papers]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    st.markdown("---")
    st.subheader("🔍 Kết quả dự đoán:")

    if prediction[0] == 2:
        st.success("🎉 Học sinh có học lực **Giỏi** ⭐")
    elif prediction[0] == 1:
        st.info("📘 Học sinh có học lực **Khá**")
    else:
        st.warning("⚠️ Học sinh có học lực **Trung bình** hoặc **Yếu**")

    st.markdown("---")

# -----------------------------
# Footer
# -----------------------------
st.caption("✅ Ứng dụng chạy trên nền Streamlit – tương thích tốt với máy tính và điện thoại.")

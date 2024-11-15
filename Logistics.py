# Import thư viện cần thiết
import pandas as pd #thư viện pandas giúp xử lý dữ liệu dạng bảng
from sklearn.model_selection import train_test_split #thư viện scikit-learn giúp chia dữ liệu thành 2 phần: huấn luyện và kiểm tra
from sklearn.preprocessing import StandardScaler #thư viện scikit-learn giúp chuẩn hóa dữ liệu
from sklearn.linear_model import LogisticRegression #thư viện scikit-learn giúp xây dựng mô hình hồi quy logistic
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #thư viện scikit-learn giúp đánh giá mô hình

# Giả sử chúng ta có bộ dữ liệu về thời tiết với các đặc trưng như nhiệt độ, độ ẩm, áp suất
data = {
    'temperature': [30, 25, 28, 35, 32, 20, 22, 30, 33, 28],
    'humidity': [70, 80, 75, 65, 60, 85, 90, 80, 75, 70],
    'pressure': [1012, 1010, 1011, 1015, 1013, 1012, 1014, 1010, 1013, 1011],
    'rain': [1, 1, 1, 0, 0, 1, 1, 0, 0, 1],  # 1 là mưa, 0 là không mưa
    'storm': [1, 0, 1, 0, 0, 1, 1, 0, 0, 1]  # 1 là có giông bão, 0 là không có giông bão
}

# Chuyển đổi dữ liệu thành DataFrame
df = pd.DataFrame(data)

# Đặc trưng và nhãn
X = df[['temperature', 'humidity', 'pressure']]  # X là các đặc trưng cần dự đoán ví dụ: Nhiệt độ, Độ ẩm, Áp suất
y_rain = df['rain']  # Y_rain là nhãn mưa dùng để dự đoán có mưa hay không
y_storm = df['storm']  # Y_storm là nhãn giông bão dùng để dự đoán có giông bão hay không

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train_rain, y_test_rain = train_test_split(X, y_rain, test_size=0.2, random_state=42)
# test_size=0.2: Chia dữ liệu thành 2 phần: 80% dữ liệu để huấn luyện và 20% dữ liệu để kiểm tra
X_train, X_test, y_train_storm, y_test_storm = train_test_split(X, y_storm, test_size=0.2, random_state=42)
# random_state=42: Để cố định dữ liệu mỗi lần chạy, giúp kết quả giống nhau mỗi lần chạy

# Chuẩn hóa dữ liệu
scaler = StandardScaler() # Khởi tạo đối tượng chuẩn hóa dữ liệu
X_train_scaled = scaler.fit_transform(X_train) # Chuẩn hóa dữ liệu huấn luyện
X_test_scaled = scaler.transform(X_test) # Chuẩn hóa dữ liệu kiểm tra

# Mô hình hồi quy logistic cho dự đoán mưa
model_rain = LogisticRegression() # Khởi tạo mô hình hồi quy logistic
model_rain.fit(X_train_scaled, y_train_rain) # Huấn luyện mô hình

# Dự đoán và đánh giá mô hình mưa
y_pred_rain = model_rain.predict(X_test_scaled)
print("Dự đoán khả năng mưa:")
print(f"Accuracy: {accuracy_score(y_test_rain, y_pred_rain)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test_rain, y_pred_rain)}")
print(f"Classification Report:\n{classification_report(y_test_rain, y_pred_rain)}")

# Mô hình hồi quy logistic cho dự đoán giông bão
model_storm = LogisticRegression()
model_storm.fit(X_train_scaled, y_train_storm)

# Dự đoán và đánh giá mô hình giông bão
y_pred_storm = model_storm.predict(X_test_scaled)
print("\nDự đoán khả năng giông bão:")
print(f"Accuracy: {accuracy_score(y_test_storm, y_pred_storm)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test_storm, y_pred_storm)}")
print(f"Classification Report:\n{classification_report(y_test_storm, y_pred_storm)}")

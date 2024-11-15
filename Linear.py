import pandas as pd #thư viện pandas giúp xử lý dữ liệu dạng bảng
from sklearn.linear_model import LinearRegression #thư viện scikit-learn giúp xây dựng mô hình hồi quy tuyến tính
import matplotlib.pyplot as plt #thư viện matplotlib giúp vẽ đồ thị
from sklearn.model_selection import train_test_split #thư viện scikit-learn giúp chia dữ liệu thành 2 phần: huấn luyện và kiểm tra

# 1. Tạo dữ liệu mẫu giả lập (có thể thay thế bằng dữ liệu thực tế)
data = {
    'Temperature': [30, 32, 35, 28, 22, 40, 25, 23, 34, 36],  # Nhiệt độ
    'Humidity': [80, 75, 85, 90, 60, 50, 65, 70, 80, 75],     # Độ ẩm
    'Wind_Speed': [15, 20, 25, 10, 5, 30, 20, 15, 12, 18],     # Tốc độ gió
    'Feels_Like': [35, 38, 40, 30, 25, 45, 30, 28, 38, 40]      # Cảm giác như thế nào
}

# Chuyển thành DataFrame
df = pd.DataFrame(data)

# 2. Chia dữ liệu thành các đặc trưng (X) và nhãn (y)
X = df[['Temperature', 'Humidity', 'Wind_Speed']]  # X là các đặc trưng cần dự đoán ví dụ: Nhiệt độ, Độ ẩm, Tốc độ gió
y = df['Feels_Like']  # Y là nhãn cần dự đoán ví dụ: Cảm giác như thế nào

# 3. Chia tập dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# test_size=0.3: Chia dữ liệu thành 2 phần: 70% dữ liệu để huấn luyện và 30% dữ liệu để kiểm tra
# random_state=42: Để cố định dữ liệu mỗi lần chạy, giúp kết quả giống nhau mỗi lần chạy


# 4. Khởi tạo và huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# 6. Hiển thị kết quả dự đoán
print("Giá trị thực tế của cảm giác: ", y_test.values)
print("Giá trị dự đoán: ", y_pred)

# 7. Đánh giá mô hình (Sử dụng R-squared để đánh giá độ chính xác)
print("R-squared: ", model.score(X_test, y_test))

# 8. Vẽ đồ thị biểu diễn kết quả (cảm giác vs nhiệt độ)
plt.scatter(y_test, y_pred)
plt.xlabel('Giá trị thực tế của cảm giác')
plt.ylabel('Giá trị dự đoán của cảm giác')
plt.title('Hồi quy tuyến tính - Cảm giác như thế nào')
plt.show()

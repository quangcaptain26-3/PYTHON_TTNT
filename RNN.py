import numpy as np # Numpy là thư viện toán học cơ bản trong Python
import pandas as pd # Pandas là thư viện xử lý dữ liệu dạng bảng
import matplotlib.pyplot as plt # Matplotlib là thư viện vẽ đồ thị
from sklearn.preprocessing import MinMaxScaler # MinMaxScaler giúp chuẩn hóa dữ liệu
from tensorflow.keras.models import Sequential # Sequential cho phép chúng ta xây dựng mô hình một cách tuần tự
from tensorflow.keras.layers import LSTM, Dense # LSTM và Dense là các lớp của mạng LSTM và mạng nơ-ron

# 1. Chuẩn bị dữ liệu
# Giả sử chúng ta có dữ liệu nhiệt độ hàng giờ trong 365 ngày
np.random.seed(42)
data = pd.DataFrame({'temperature': 20 + 10 * np.sin(np.linspace(0, 2 * np.pi, 365*24)) + np.random.normal(0, 1, 365*24)}) # Tạo dữ liệu nhiệt độ
print(data.head())

# 2. Tiền xử lý dữ liệu
# Chuẩn hóa dữ liệu trong khoảng [0, 1] để phù hợp với LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
data['temperature_scaled'] = scaler.fit_transform(data[['temperature']])

# Định nghĩa hàm để tạo chuỗi dữ liệu đầu vào cho LSTM
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

# Sử dụng 24 giờ trước để dự đoán nhiệt độ của giờ tiếp theo
seq_length = 24
X, y = create_sequences(data['temperature_scaled'].values, seq_length)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Chuyển đổi dữ liệu thành dạng mà LSTM có thể sử dụng (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 3. Xây dựng mô hình LSTM
model = Sequential([
    LSTM(50, input_shape=(seq_length, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 4. Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# 5. Dự đoán
y_pred = model.predict(X_test)

# Chuyển đổi dự đoán về giá trị ban đầu
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Đánh giá mô hình bằng cách vẽ biểu đồ kết quả dự đoán so với giá trị thực tế
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, label='Giá trị thực')
plt.plot(y_pred_rescaled, label='Dự đoán', linestyle='--')
plt.xlabel('Thời gian')
plt.ylabel('Nhiệt độ')
plt.legend()
plt.show()

# 6. Đánh giá mô hình
loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Loss: {loss}')





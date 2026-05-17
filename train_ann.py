import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
df = pd.read_csv('ran_data.csv')

# Features and target
X = df[['Frequency', 'Bandwidth', 'SINR', 'TxPower']]
y = df['Throughput']

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build ANN model
model = Sequential([
    Dense(10, activation='relu', input_shape=(4,)),
    Dense(8, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_scaled, y, epochs=300, verbose=0)

# Predict on training data
predictions = model.predict(X_scaled)
print(predictions)
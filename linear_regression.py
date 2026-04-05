import pandas as pd
import matplotlib.pyplot as plt 

# 1. Load data
data = pd.read_csv('train.csv')
cols = ['open', 'high', 'liquidity_ratio', 'low', 'beta_indicator', 'close']
data = data.dropna(subset=cols)

# Total error / loss function - Refactored for speed
def loss(m1, m2, m3, m4, m5, b, points):
    # Instead of a loop, we calculate the entire column at once
    prediction = (m1*points.open + m2*points.high + m3*points.liquidity_ratio + 
                  m4*points.low + m5*points.beta_indicator + b)
    
    total_error = ((points.close - prediction) ** 2).mean()
    return total_error

def gradient_descent(m1_current, m2_current, m3_current, m4_current, m5_current, b_current, l, points):
    n = len(points)
    
    # Calculate predictions for all rows at once
    prediction = (m1_current*points.open + m2_current*points.high + 
                  m3_current*points.liquidity_ratio + m4_current*points.low + 
                  m5_current*points.beta_indicator + b_current)
    
    # Calculate error vector
    error = points.close - prediction

    # Calculate gradients using column-wise multiplication (Vectorization)
    m1_gradient = -(2/n) * (points.open * error).sum()
    m2_gradient = -(2/n) * (points.high * error).sum()
    m3_gradient = -(2/n) * (points.liquidity_ratio * error).sum()
    m4_gradient = -(2/n) * (points.low * error).sum()
    m5_gradient = -(2/n) * (points.beta_indicator * error).sum()
    b_gradient  = -(2/n) * error.sum()

    # Update weights
    m1 = m1_current - l * m1_gradient
    m2 = m2_current - l * m2_gradient
    m3 = m3_current - l * m3_gradient
    m4 = m4_current - l * m4_gradient
    m5 = m5_current - l * m5_gradient
    b  = b_current - l * b_gradient
    
    return m1, m2, m3, m4, m5, b

# Initialize variables
m1=m2=m3=m4=m5=b=0
l = 0.000001 # Note: Keep learning rate very small if data isn't scaled
epochs = 100

print("Training...")
for i in range(epochs):
    m1, m2, m3, m4, m5, b = gradient_descent(m1, m2, m3, m4, m5, b, l, data)
    if i % 10 == 0:
        print(f"Epoch {i} completed")

print("\nTraining Complete ✅")
print(f"Final Weights:\nm1: {m1}\nm2: {m2}\nm3: {m3}\nm4: {m4}\nm5: {m5}\nb: {b}")

# --- IMPROVED GRAPH PART ---
plt.figure(figsize=(10, 6))

# Calculate predictions
predictions = (m1*data.open + m2*data.high + m3*data.liquidity_ratio + 
               m4*data.low + m5*data.beta_indicator + b)

# Plot Actual vs Predicted
plt.scatter(data.close, predictions, color="blue", alpha=0.3)

# Add a 45-degree line (Perfect Prediction Line)
ideal_line = [data.close.min(), data.close.max()]
plt.plot(ideal_line, ideal_line, color="red", linestyle="--", label="Perfect Prediction")

plt.xlabel("Actual Close Price")
plt.ylabel("Predicted Close Price")
plt.title("Model Accuracy: Actual vs. Predicted")
plt.legend()
plt.show()
#plt.savefig('accuracy_plot.png')
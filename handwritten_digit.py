import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# -------------------------------------------------
# 1. Load MNIST Dataset
# -------------------------------------------------
print("Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training images:", x_train.shape)
print("Testing images :", x_test.shape)

# -------------------------------------------------
# 2. Preprocess the Data
# -------------------------------------------------
# Convert pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# -------------------------------------------------
# 3. Display Sample Images
# -------------------------------------------------
plt.figure(figsize=(10, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Digit: {y_train[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# -------------------------------------------------
# 4. Build Neural Network
# -------------------------------------------------
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# -------------------------------------------------
# 5. Compile Model
# -------------------------------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display model structure
model.summary()

# -------------------------------------------------
# 6. Train the Model
# -------------------------------------------------
print("\nTraining the model...")

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

# -------------------------------------------------
# 7. Evaluate Model
# -------------------------------------------------
print("\nEvaluating the model...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# -------------------------------------------------
# 8. Plot Training Accuracy
# -------------------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.show()

# -------------------------------------------------
# 9. Make Predictions
# -------------------------------------------------
predictions = model.predict(x_test)

# -------------------------------------------------
# 10. Display Predictions
# -------------------------------------------------
plt.figure(figsize=(12, 6))

for i in range(10):
    predicted_digit = np.argmax(predictions[i])

    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], cmap="gray")
    plt.title(
        f"Actual: {y_test[i]}\nPredicted: {predicted_digit}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()

print("\nProject completed successfully!")

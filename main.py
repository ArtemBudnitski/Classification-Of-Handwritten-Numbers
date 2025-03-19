import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageDraw


def load_data():
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, Y_train, X_test, Y_test


def show_sample_images(X_train, Y_train, num_images=10):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_train[i], cmap="gray")
        plt.title(f"Label: {Y_train[i]}")
        plt.axis('off')
    plt.show()


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(62, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))
    return model


def evaluate_model(model, X_test, Y_test):
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_acc}")


def show_prediction(model, X_test, index=0):
    predictions = model.predict(X_test)
    prediction_label = np.argmax(predictions[index])
    plt.imshow(X_test[index], cmap="gray")
    plt.title(f"Prediction: {prediction_label}")
    plt.axis('off')
    plt.show()


class DigitRecognizerApp:
    def __init__(self, model):
        self.model = model
        self.window = tk.Tk()
        self.window.title("Handwritten Digit Recognition")

        self.canvas = tk.Canvas(self.window, width=280, height=280, background="white")
        self.canvas.pack()

        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Predict button
        self.button_predict = tk.Button(self.window, text="Predict", command=self.predict_digit, height=2, width=2)
        self.button_predict.pack()

        # Clear button
        self.button_clear = tk.Button(self.window, text="Clear", command=self.clear_canva, height=2, width=2)
        self.button_clear.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.window.mainloop()

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def clear_canva(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

    def preprocess_image(self):
        img = self.image.resize((28, 28))
        img = img.convert("L")
        img = np.array(img)

        img = 255 - img
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28)
        return img

    def predict_digit(self):
        processed_image = self.preprocess_image()
        prediction = self.model.predict(processed_image)
        predicted_label = np.argmax(prediction)

        result_label = tk.Label(self.window, text=f"{predicted_label}", font=("Arial", 20))
        result_label.pack()

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_data()
    # show_sample_images(X_train, Y_train)
    model = build_model()
    train_model(model, X_train, Y_train, X_test, Y_test)
    evaluate_model(model, X_test, Y_test)

    # show_prediction(model, X_test, 3)
    DigitRecognizerApp(model)

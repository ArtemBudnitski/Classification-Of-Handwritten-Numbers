# 🖊️ Classification of Handwritten Numbers

This project is a deep learning-based application for recognizing handwritten digits using the MNIST dataset. The application allows users to train a neural network model and test predictions with an interactive graphical interface.

## 📌 Features
- Loads and preprocesses the MNIST dataset.
- Trains a neural network for handwritten digit classification.
- Evaluates the model's performance.
- Provides an interactive GUI for drawing and predicting handwritten digits.

## 🛠️ Technologies Used
- TensorFlow/Keras
- NumPy
- Matplotlib
- Tkinter
- Pillow (PIL)

## 🚀 Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ArtemBudnitski/Classification-Of-Handwritten-Numbers
   cd ClassificationOfHandwrittenNumbers
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python main.py
   ```

## 📊 Model Overview
The model consists of:
- **Input Layer:** 28x28 flattened grayscale image
- **Hidden Layer:** 128 neurons with ReLU activation and dropout (0.2)
- **Output Layer:** 62 neurons with softmax activation (for digit classification)

## 🎨 GUI Usage
1. Draw a digit in the provided canvas.
2. Click **"Predict"** to classify the drawn digit.
3. Click **"Clear"** to reset the canvas.

---

# 🖊️ Klasyfikacja Ręcznie Pisanych Liczb

Ten projekt to aplikacja oparta na uczeniu głębokim do rozpoznawania ręcznie pisanych cyfr przy użyciu zbioru danych MNIST. Użytkownik może trenować model sieci neuronowej oraz testować predykcje w interfejsie graficznym.

## 📌 Funkcjonalności
- Ładowanie i wstępne przetwarzanie zbioru danych MNIST.
- Trenowanie modelu sieci neuronowej do klasyfikacji cyfr.
- Ocena wydajności modelu.
- Interaktywny interfejs GUI do rysowania i klasyfikacji cyfr.

## 🛠️ Wykorzystane Technologie
- TensorFlow/Keras
- NumPy
- Matplotlib
- Tkinter
- Pillow (PIL)

## 🚀 Instalacja i konfiguracja

1. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/ArtemBudnitski/Classification-Of-Handwritten-Numbers
   cd ClassificationOfHandwrittenNumbers
   ```

2. Utwórz i aktywuj wirtualne środowisko (opcjonalne, ale zalecane):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. Zainstaluj wymagane zależności:
   ```bash
   pip install -r requirements.txt
   ```

4. Uruchom aplikację:
   ```bash
   python main.py
   ```

## 📊 Przegląd modelu
Model składa się z:
- **Warstwa wejściowa:** 28x28 spłaszczony obraz w skali szarości.
- **Warstwa ukryta:** 128 neuronów z aktywacją ReLU oraz dropout (0.2).
- **Warstwa wyjściowa:** 62 neurony z aktywacją softmax (klasyfikacja cyfr).

## 🎨 Obsługa GUI
1. Narysuj cyfrę na dostępnej planszy.
2. Kliknij **"Predict"**, aby sklasyfikować cyfrę.
3. Kliknij **"Clear"**, aby wyczyścić rysunek.

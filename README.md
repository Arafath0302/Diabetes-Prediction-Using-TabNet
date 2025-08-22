# 🩺 Diabetes Prediction App (TabNet)

This project is a **machine learning application** to predict diabetes based on patient health indicators.  
It includes both the **Streamlit frontend** (`app.py`) for interactive prediction and the **training script** (`train.py`) showing how the model was built.  

---

## 📂 Project Structure
```
diaBetes_Predict/
│── app.py              # Streamlit frontend
│── train.py            # Training script (for reference/retraining)
│── model.pkl           # Pre-trained model (used by app.py)
│── requirements.txt    # Project dependencies
│── README.md           # Project documentation
```

---

## 🚀 Running the App

1. **Clone the repository**  
   ```bash
   git clone https://github.com/<your-username>/diaBetes_Predict.git
   cd diaBetes_Predict
   ```

2. **Create a virtual environment** (recommended)  
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**  
   ```bash
   streamlit run app.py
   ```

The app will open in your browser (default: `http://localhost:8501`).  

---

## 🧠 Model Training (Optional)

The training script (`train.py`) is provided for **reference** to show how `model.pkl` was created.  
It includes:
- Data preprocessing (encoding, normalization)  
- Handling class imbalance with **SMOTE**  
- Training a **TabNet classifier**  
- Saving the trained pipeline as `model.pkl`  

To retrain:
```bash
python train.py
```

---

## 📊 Features

- Age, gender, blood pressure, glucose, BMI, and other health indicators  
- Balanced training data (SMOTE)  
- TabNet deep learning model  
- Feature preprocessing pipeline included in `model.pkl`  
- Interactive web app powered by **Streamlit**  

---

## 🛠 Requirements

Main dependencies:
- Python 3.10+  
- Streamlit  
- Scikit-learn  
- Imbalanced-learn  
- PyTorch TabNet  
- Joblib  
- Pandas / Numpy  

All packages are listed in `requirements.txt`.  

---

## 📌 Notes
- `model.pkl` is already included, so **you don’t need to retrain** to use the app.  
- `train.py` is only for demonstration and reproducibility.  
- If you face issues with version mismatches (e.g., scikit-learn), ensure you are using the versions in `requirements.txt`.  

---



## 📜 License
This project is released under the **MIT License**.  

---

## 📦 requirements.txt

```txt
streamlit==1.48.1
pandas==2.3.1
numpy==2.3.2
scikit-learn==1.4.2
torch==2.8.0
pytorch-tabnet==4.1.0
imbalanced-learn==0.12.3
joblib==1.4.2
```

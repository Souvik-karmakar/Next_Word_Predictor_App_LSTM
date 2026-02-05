# 🧠 Next Word Prediction Using LSTM

## 📌 Project Overview
This project implements a **Next Word Prediction model** using a **Long Short-Term Memory (LSTM)** neural network to predict the most probable next word in a given text sequence. The model is trained on **Shakespeare’s _Hamlet_**, enabling it to learn complex linguistic patterns and contextual relationships.

The project covers the complete **end-to-end NLP pipeline**, including data collection, preprocessing, model training with early stopping, evaluation, and deployment using **Streamlit** for real-time predictions.

---

## 🎯 Objective
- Predict the next word in a given sentence using sequence modeling  
- Learn contextual word relationships using LSTM-based RNNs  
- Deploy an interactive web application for real-time inference  

---

## 📂 Dataset
- **Source:** NLTK Gutenberg Corpus  
- **Text:** Shakespeare – _Hamlet_  
- The dataset provides rich vocabulary and complex sentence structures for effective language modeling.

---
## App Screenshot
![App Screenshot](LSTM_Next_Word_Prediction_App.png)

---

## ⚙️ Project Workflow

### 1️⃣ Data Collection
- Loaded Shakespeare’s *Hamlet* using NLTK’s Gutenberg corpus  
- Stored raw text locally for preprocessing  

### 2️⃣ Data Preprocessing
- Converted text to lowercase  
- Tokenized text using **Keras Tokenizer**  
- Generated n-gram sequences  
- Applied padding to ensure uniform input length  
- Split data into training and testing sets  

---

### 3️⃣ Model Architecture

#### 🔹 LSTM Model
- Embedding Layer  
- Two stacked LSTM layers  
- Dropout for regularization  
- Dense layer with Softmax activation  

#### 🔹 GRU Model (Experimental)
- Implemented GRU-based RNN for comparison with LSTM  

---

### 4️⃣ Model Training
- **Loss Function:** Categorical Crossentropy  
- **Optimizer:** Adam  
- **Metric:** Accuracy  
- **Early Stopping** applied to prevent overfitting by monitoring validation loss  

---

### 5️⃣ Model Evaluation
- Tested the model using unseen text sequences  
- Predicted next words based on learned contextual patterns  

---

### 6️⃣ Deployment
- Built an interactive **Streamlit application**  
- Users can input a sequence of words and receive real-time predictions  
- Model and tokenizer are loaded for inference
- Live App Link :- https://nextwordpredictorapplstm-b48qlgs2lekfte4pqfmzbb.streamlit.app/

---

## 🧪 Example Prediction
```text
Input:  To be or not to be
Output: that

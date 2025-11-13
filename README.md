# 🚀 Optimization Algorithms, Image Reconstruction & Logistic Regression

This project explores core machine learning concepts through hands-on implementation of optimization algorithms, time-series forecasting, matrix factorization for image reconstruction and compression, and a custom logistic regression model built from scratch using PyTorch.

---

## 👥 Team Members
- **Sai Gawali**
- **Siddharth Nayak**
- **Siddhesh Patil**

---

## 📌 Overview

This project brings together several fundamental ML workflows:

- Gradient Descent variants (Full-batch, Stochastic, Momentum)
- Effect of feature scaling on optimization
- Autoregressive forecasting using linear models
- Low-rank matrix factorization (GD + ALS)
- Image reconstruction & compression
- Custom Logistic Regression using PyTorch
- Visualization of optimization trajectories, loss curves, decision boundaries, and reconstruction quality

The goal was to deeply understand how optimization behaves in practice and how low-rank representations and classical models perform across tasks.

---

# 🧠 1. Optimization Algorithms

### 🔹 Implemented
- Full-batch Gradient Descent  
- Stochastic Gradient Descent  
- Gradient Descent with Momentum  

### 🔹 Key Insights
- Compared convergence speed across datasets  
- Visualized:
  - Optimization paths on contour plots  
  - Loss vs epochs  
  - Effect of momentum vector updates  
- Studied how feature scaling dramatically improves optimization stability and speed

---

# 🌡️ 2. Feature Scaling & Convergence

Analyzed gradient descent behavior on a dataset with large-scale features (0–1000).

### 🔹 Highlights
- Showed slow and unstable convergence without scaling  
- Applied **z-score normalization**  
- Demonstrated significant improvement in iteration count and loss curve smoothness

---

# 📈 3. Autoregressive Time-Series Forecasting

Built an AR(1) model to predict next-day temperature using historical daily temperature data.

### 🔹 Results
- Linear regression-based autoregressive forecasting  
- Plotted predicted vs true temperatures  
- Reported RMSE and discussed forecasting limitations  

---

# 🖼️ 4. Matrix Factorization (Image Reconstruction & Compression)

## 🔧 4.1 Image Reconstruction

Performed low-rank matrix factorization to recover missing pixels.

### 🔹 Tasks
- Reconstructed images with:
  - A missing **30×30 rectangular region**  
  - **900 random missing pixels**  
- Implemented:
  - Gradient Descent–based factorization  
  - Alternating Least Squares (ALS)  

### 🔹 Metrics
- RMSE between reconstructed and original image  
- Peak Signal-to-Noise Ratio (PSNR)  

### 🔹 Visualizations
- Original vs masked vs reconstructed images

---

## 📦 4.2 Image Compression

Explored low-rank factorization as a compression technique on 50×50 image patches.

### 🔹 Experiments
- Tested patches with:
  - 1 dominant color  
  - 2–3 colors  
  - 5+ colors  
- Used ranks: **5, 10, 25, 50**  
- Reconstructed compressed patches and compared visual quality

---

# 🤖 5. Logistic Regression in PyTorch

A complete logistic regression classifier built **from scratch** in PyTorch without using scikit-learn.

### 🔹 Features
- Custom class with:
  - `fit()`
  - `predict()`
  - `predict_proba()`
- Trained on **make_moons** dataset  

### 🔹 Evaluation
- Decision boundary visualizations  
- Loss curve during training  
- Accuracy comparison with `sklearn.linear_model.LogisticRegression`  

---

# 🛠️ Tech Stack
- **Python**
- **NumPy**
- **Matplotlib**
- **PyTorch**
- **Scikit-learn**

---

# 🎯 Summary

This project demonstrates:
- Strong understanding of optimization algorithms  
- Practical experience with time-series modeling  
- Deep knowledge of matrix factorization (GD + ALS)  
- Ability to build ML models from scratch in PyTorch  
- Strong visualization and evaluation of ML behavior  

---

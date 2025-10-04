# 🧠 Machine Learning Progression — From Logistic Regression to Neural Networks

This repository documents my journey through foundational machine learning concepts — starting from simple classification using **logistic regression** and progressing towards **multilayer perceptron (MLP)** neural networks.  

---

## 📘 Contents

### 1. `basicdemo1.ipynb` — Logistic Regression (Simple Demo)
**Objective:** Build a basic classification model using the Iris dataset.

**What I did:**
- Imported and explored the classic `iris.csv` dataset.
- Split the features (`X`) and labels (`Y`).
- Trained a `LogisticRegression` model from `scikit-learn`.
- Made predictions on new flower data.
- Printed and interpreted the predicted classes.

**Lessons Learned:**
- The relationship between input features and categorical outputs.
- How logistic regression classifies linearly separable data.
- Importance of understanding feature-label mappings.
- How to quickly make predictions on unseen data.

---

### 2. `basicdemo2.ipynb` — Logistic Regression with Preprocessing & Evaluation
**Objective:** Improve the classification workflow with better preprocessing and evaluation.

**What I did:**
- Used `train_test_split()` to separate training and testing data.
- Standardized features using `StandardScaler` to ensure all variables are on the same scale.
- Trained and tested the logistic regression model.
- Calculated accuracy using `accuracy_score`.
- Predicted new samples using scaled input data.

**Lessons Learned:**
- **Data preprocessing** is crucial — models perform better when features are standardized.
- Splitting datasets prevents overfitting and tests real-world performance.
- Learned to evaluate a model using quantitative metrics.
- Practiced end-to-end ML workflow (load → preprocess → train → evaluate → predict).

---

### 3. `classification (1).ipynb` — Interactive Neural Network Classification
**Objective:** Explore nonlinear decision boundaries using a neural network (MLP).

**What I did:**
- Generated circular synthetic data using `make_circles()` from `sklearn.datasets`.
- Built an interactive visualization using:
  - `MLPClassifier` (multi-layer perceptron)
  - `ipywidgets` sliders for adjusting hidden layer size
  - `matplotlib` contour plots for visualizing decision boundaries
- Observed how increasing the hidden layer size changes model complexity.

**Lessons Learned:**
- Neural networks can learn **nonlinear** decision boundaries that logistic regression cannot.
- Increasing the number of hidden neurons allows the network to fit more complex shapes.
- How to visualize classification results in 2D space using contour plots.
- Gained intuition for **activation functions** and **hidden layers**.

---

## 🚀 Next Step — Email Spam Detector (Planned Implementation)
**Goal:** Apply everything I’ve learned to a **real-world text classification problem**.

**Planned Steps:**
1. **Dataset:** Use the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) (or a similar public dataset).
2. **Preprocessing:** Convert text into numerical features using `TfidfVectorizer`.
3. **Model:** Train an `MLPClassifier` to detect spam emails.
4. **Evaluation:** Visualize confusion matrices, accuracy, and top indicative words.
5. **Visualization Ideas:**
   - PCA projection of TF-IDF vectors.
   - t-SNE visualization of predicted classes.
   - Loss curve tracking over training iterations.

**Concepts I’ll Practice:**
- Working with **text data** and **feature extraction**.
- Understanding **how MLPs handle nonlinearity in feature space**.
- Interpreting and visualizing model performance.

---

## 🧩 Summary of Concepts Covered
| Concept | Learned In | Description |
|----------|-------------|-------------|
| Logistic Regression | `basicdemo1`, `basicdemo2` | Foundation of classification models |
| Data Preprocessing | `basicdemo2` | Scaling and splitting data |
| Model Evaluation | `basicdemo2` | Accuracy and test/train split |
| Neural Networks (MLP) | `classification (1).ipynb` | Learning nonlinear patterns |
| Visualization | `classification (1).ipynb` | Understanding decision boundaries |
| Future Work | — | Text-based MLP classification (spam detection) |

---

## 🛠️ Tools Used
- **Python 3**
- **scikit-learn**
- **matplotlib**
- **numpy**
- **pandas**
- **ipywidgets** (for interactivity)

# ❤️ Heart Disease ML Project

## 📝 Project Overview

This project aims to **analyze** and **predict heart disease risk** using the **UCI Heart Disease dataset**.

It includes:

* 🧹 **Data preprocessing & cleaning**
* 📊 **Feature selection & dimensionality reduction (PCA)**
* 🤖 **Supervised learning**
  (Logistic Regression, Decision Tree, Random Forest, SVM)
* 🧩 **Unsupervised learning**
  (K-Means, Hierarchical Clustering)
* ⚙️ **Model optimization** (hyperparameter tuning)
* 🌐 **Streamlit web UI** for real-time predictions
* 🚀 **[Bonus] Deployment via Ngrok**

---

## 📁 Folder Structure

```text
Heart_Disease_Project/
│── data/
│   ├── heart_disease.csv
│   └── heart_disease_selected_features.csv
│
│── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│
│── models/
│   ├── Random_Forest_model.pkl
│   ├── Logistic_Regression_model.pkl
│   ├── Decision_Tree_model.pkl
│   ├── SVM_model.pkl
│   ├── scaler.pkl
│
│── UI/
│   └── app.py
│
│── Deploy/
│   └── ngrok_deploy.py
│
│── results/
│   └── evaluation_metrics.txt
│
│── README.md
│── requirements.txt
│── .gitignore
```

---

## ⚙️ Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

**Main libraries used:**

* `pandas`, `numpy`
* `matplotlib`, `seaborn`
* `scikit-learn`
* `streamlit`
* `pyngrok`
* `joblib`

---

## 🚀 How to Run the Project

### 1️⃣ Run the Jupyter Notebooks

Open each notebook in the `notebooks/` folder to:

* Perform data preprocessing & cleaning
* Apply PCA and dimensionality reduction
* Perform feature selection
* Train supervised & unsupervised models
* Run hyperparameter tuning

You can use:

```bash
jupyter notebook
```

and then open the notebooks from the browser.

---

### 2️⃣ Run the Streamlit UI

From the project root, navigate to the UI folder:

```bash
cd Heart_Disease_Project/UI
streamlit run app.py
```

Then:

* Open the URL shown in the terminal (usually `http://localhost:8501`)
* Enter patient data in the web interface
* Get **real-time heart disease predictions**

---

### 3️⃣ Ngrok Deployment [Bonus]

1. Create and verify your **Ngrok** account and get your `AUTHTOKEN`.
2. Authenticate Ngrok:

```bash
ngrok authtoken <YOUR_AUTHTOKEN>
```

3. From the `Deploy/` folder, run:

```bash
cd Heart_Disease_Project/Deploy
python ngrok_deploy.py
```

4. Copy the **public Ngrok URL** shown in the console and open it in your browser to access the Streamlit app online.

---

## 💾 Model Files

The following trained models are stored in the `models/` folder:

* `Random_Forest_model.pkl` – trained Random Forest classifier
* `Logistic_Regression_model.pkl` – trained Logistic Regression model
* `Decision_Tree_model.pkl` – trained Decision Tree classifier
* `SVM_model.pkl` – trained Support Vector Machine classifier
* `scaler.pkl` – `StandardScaler` used for data preprocessing

Make sure these files stay in the `models/` directory so the app can load them correctly.

---

## 📊 Data Files

Located in the `data/` folder:

* `heart_disease.csv` – original UCI heart disease dataset
* `heart_disease_selected_features.csv` – dataset with selected key features after preprocessing/feature selection

---

## 📈 Results

Evaluation metrics for all models are stored in:

```text
results/evaluation_metrics.txt
```

This includes:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Use these metrics to compare model performance and select the best one for deployment.

---

## 📝 Notes & Tips

* Use **absolute or project-relative paths** in `app.py` when loading models and data to avoid `FileNotFoundError`.

* Always run Streamlit apps with:

  ```bash
  streamlit run app.py
  ```

* If you modify the models or retrain them, don’t forget to:

  * Save them again to the `models/` folder
  * Update any paths or preprocessing steps accordingly

---

## 🔮 Future Work

* Add **Deep Learning models** such as ANN or CNN for improved performance.
* Integrate **real hospital data** instead of only relying on the UCI dataset.
* Add **model explainability** using SHAP or LIME.
* Improve the **UI design** with better visualization and patient reports.
* Deploy the system on a **cloud platform** (Heroku, Render, or AWS).

---

## ⚠️ Limitations

* The dataset is relatively **small** and may not fully represent real-world cases.
* Predictions depend heavily on the **quality of input features**.
* The system should be used for **educational purposes only**, not as a medical diagnosis tool.

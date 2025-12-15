import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = r'D:\Heart_Disease_Project\Heart_Disease_Project\data\heart_disease_selected_features.csv'
MODEL_PATHS = {
    "Random Forest": r'D:\Heart_Disease_Project\Heart_Disease_Project\models\Random_Forest_model.pkl',
    "Logistic Regression": r'D:\Heart_Disease_Project\Heart_Disease_Project\models\Logistic_Regression_model.pkl',
    "Decision Tree": r'D:\Heart_Disease_Project\Heart_Disease_Project\models\Decision_Tree_model.pkl',
    "SVM": r'D:\Heart_Disease_Project\Heart_Disease_Project\models\SVM_model.pkl'
}
RESULTS_PATH = r'D:\Heart_Disease_Project\Heart_Disease_Project\results\evaluation_metrics.txt'

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)
X = df.drop('target', axis=1)
y = df['target']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Load models
# -----------------------------
models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}

# -----------------------------
# Evaluate and save metrics
# -----------------------------
with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
    f.write("# Heart Disease ML Models Evaluation Metrics\n\n")
    
    for i, (name, model) in enumerate(models.items(), 1):
        y_pred = model.predict(X_test)
        
        # ROC-AUC requires probability scores
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        except:
            roc_auc = "N/A"
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        f.write(f"{i}️⃣ {name}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc}\n\n")
    
    f.write("# Notes:\n")
    f.write("- Metrics calculated on the test set (20% of the data)\n")
    f.write("- ROC-AUC computed using probability scores where available\n")

print(f"✅ Evaluation metrics saved to {RESULTS_PATH}")

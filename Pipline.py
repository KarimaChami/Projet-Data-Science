import pandas as pd 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
# data = pd.read_csv("data-68e11476082f9096032105.csv")
from sklearn.metrics import roc_curve,auc,precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline 

#Fonctions de préparation des données'
def load_data(dt):
  dt =  pd.read_csv(f"{dt}.csv")
  return dt
data = load_data("data")
# print(data.head())


def Encodage(data):
    data = data.drop(columns=['customerID','gender'])
    label_oncoder  =  LabelEncoder()
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        data[col] = label_oncoder.fit_transform(data[col])

    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data = data.fillna(data['TotalCharges'].mean())
    return data

data = Encodage(data)



def split_data(data, target='Churn'):
    X = data.drop(columns=[target])
    y = data[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)
# split_data()
X_train, X_test, y_train, y_test = split_data(data)

def train_models(X_train, y_train):
    models = {
        'Logistic Regression': Pipeline([
            ('scaler',StandardScaler()),
            ('model',LogisticRegression())
        ]),
        'Random Forest': RandomForestClassifier()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_pred)
    }
trained_models = train_models(X_train, y_train)



# from sklearn.preprocessing import StandardScaler #méthode de normalisation
# def Normalisation(X_train,X_test):
#    scaler = StandardScaler()
#    X_train_scaled = scaler.fit_transform(X_train)
#    X_test_scaled = scaler.transform(X_test)
#    return X_train_scaled,X_test_scaled
# Normalisation()

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

# trained_models = train_models(X_train, y_train)


def afficher_courbes(model, X_test, y_test, name):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # --- ROC ---
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # --- PR ---
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    average_precision = average_precision_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('Faux positifs (FPR)')
    plt.ylabel('Vrais positifs (TPR)')
    plt.title(f'Courbe ROC - {name}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', label=f'AP = {average_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Courbe PR - {name}')
    plt.legend()
    plt.tight_layout()
    plt.show()
#Ce code ne s’exécutera que si tu lances directement python Pipline.py → Il ne s’exécutera pas quand pytest importe le fichier.
if __name__ == "__main__":  
    for name, model in trained_models.items():
        print(f"\n{name}")
        afficher_courbes(model, X_test, y_test, name)



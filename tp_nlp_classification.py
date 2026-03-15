import os
import tarfile
import urllib.request
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report, confusion_matrix)


DATASET_URL  = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
ARCHIVE_PATH = "aclImdb_v1.tar.gz"
DATA_DIR     = "aclImdb"


def download_dataset():
    if not os.path.exists(DATA_DIR):
        if not os.path.exists(ARCHIVE_PATH):
            print("Telechargement du dataset...")
            urllib.request.urlretrieve(DATASET_URL, ARCHIVE_PATH)
        print("Extraction...")
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall()
    else:
        print("Dataset deja present.")


def load_reviews(split="train"):
    texts, labels = [], []
    for label_str, label_val in [("pos", 1), ("neg", 0)]:
        folder = os.path.join(DATA_DIR, split, label_str)
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    texts.append(f.read())
                    labels.append(label_val)
    return texts, labels


def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)


def evaluate_model(model_name, y_test, y_pred):
    print(f"\n{'='*50}")
    print(f"  Resultats : {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")
    print("\nRapport detaille :")
    print(classification_report(y_test, y_pred, target_names=["Negatif", "Positif"]))


def plot_confusion_matrix(y_test, y_pred, model_name, filename):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negatif", "Positif"],
                yticklabels=["Negatif", "Positif"])
    plt.title(f"Matrice de confusion - {model_name}")
    plt.xlabel("Predit")
    plt.ylabel("Reel")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def predict_sentiment(phrase, vectorizer, model, model_name):
    phrase_clean  = preprocess(phrase)
    phrase_vector = vectorizer.transform([phrase_clean])
    prediction    = model.predict(phrase_vector)[0]
    proba         = model.predict_proba(phrase_vector)[0]
    label         = "POSITIF" if prediction == 1 else "NEGATIF"
    confiance     = proba[prediction] * 100
    print(f"[{model_name}] \"{phrase}\" => {label} ({confiance:.1f}%)")


download_dataset()

print("\nChargement des donnees...")
train_texts, train_labels = load_reviews("train")
test_texts,  test_labels  = load_reviews("test")
print(f"Train : {len(train_texts)} critiques")
print(f"Test  : {len(test_texts)} critiques")

idx_pos = train_labels.index(1)
idx_neg = train_labels.index(0)
print(f"\nExemple POSITIF : {train_texts[idx_pos][:200]}...")
print(f"\nExemple NEGATIF : {train_texts[idx_neg][:200]}...")

positifs = sum(train_labels)
negatifs = len(train_labels) - positifs
print(f"\nDistribution train => Positifs : {positifs} | Negatifs : {negatifs}")

print("\nPreprocessing...")
train_clean = [preprocess(t) for t in train_texts]
test_clean  = [preprocess(t) for t in test_texts]
print("Preprocessing termine.")

print("\nVectorisation TF-IDF...")
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_clean)
X_test  = vectorizer.transform(test_clean)
y_train = np.array(train_labels)
y_test  = np.array(test_labels)
print(f"Forme X_train : {X_train.shape}")
print(f"Forme X_test  : {X_test.shape}")

print("\nEntrainement Naive Bayes...")
nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
evaluate_model("Naive Bayes", y_test, y_pred_nb)
plot_confusion_matrix(y_test, y_pred_nb, "Naive Bayes", "confusion_matrix_nb.png")

print("\nEntrainement Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
evaluate_model("Logistic Regression", y_test, y_pred_lr)
plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression", "confusion_matrix_lr.png")

acc_nb = accuracy_score(y_test, y_pred_nb)
acc_lr = accuracy_score(y_test, y_pred_lr)
f1_nb  = f1_score(y_test, y_pred_nb)
f1_lr  = f1_score(y_test, y_pred_lr)

models     = ["Naive Bayes", "Logistic Regression"]
accuracies = [acc_nb, acc_lr]
f1_scores  = [f1_nb, f1_lr]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, accuracies, width, label="Accuracy", color="#4C72B0")
bars2 = ax.bar(x + width/2, f1_scores,  width, label="F1-Score",  color="#DD8452")

ax.set_ylim(0.8, 1.0)
ax.set_ylabel("Score")
ax.set_title("Comparaison Naive Bayes vs Logistic Regression")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

for bar in bars1 + bars2:
    ax.annotate(f"{bar.get_height():.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("comparaison_modeles.png", dpi=150)
plt.show()

print("\n" + "="*50)
print("  Test avec nouvelles phrases")
print("="*50)

test_phrases = [
    "This movie was amazing and I loved it",
    "Absolutely brilliant, one of the best films ever",
    "Terrible movie, complete waste of time",
    "Boring and awful, I fell asleep after 10 minutes",
    "Some scenes were great but overall it was quite boring",
    "Not good, not great, just completely average",
    "I could not stop watching, it was so good",
]

print("\n--- Naive Bayes ---")
for phrase in test_phrases:
    predict_sentiment(phrase, vectorizer, nb_model, "NB")

print("\n--- Logistic Regression ---")
for phrase in test_phrases:
    predict_sentiment(phrase, vectorizer, lr_model, "LR")

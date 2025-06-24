import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from sentence_transformers import SentenceTransformer
import spacy
import torch

def balancear_dataset(df, coluna_classe):
    contagens = df[coluna_classe].value_counts()
    max_count = contagens.max()
    print(f"Balanceando para {max_count} instâncias por classe")
    balanced_df = pd.concat([
        df[df[coluna_classe] == classe].sample(n=max_count, replace=True, random_state=42)
        for classe in contagens.index
    ])
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.show()

def plot_roc(y_test, y_score, classes, model_name):
    y_test_bin = label_binarize(y_test, classes=classes)
    plt.figure(figsize=(7, 6))
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.title(f'Curva ROC - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_precision_recall(y_test, y_score, classes, model_name):
    y_test_bin = label_binarize(y_test, classes=classes)
    plt.figure(figsize=(7, 6))
    for i, class_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        plt.plot(recall, precision, label=f'{class_name}')
    plt.title(f'Curva Precisão vs. Revocação - {model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def avaliar_modelo(X, y, nome_modelo):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\n=== {nome_modelo} ===")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred, model.classes_, f"{nome_modelo} - Matriz de Confusão")
    plot_roc(y_test, y_proba, model.classes_, nome_modelo)
    plot_precision_recall(y_test, y_proba, model.classes_, nome_modelo)

    return {
        'model': nome_modelo,
        'accuracy': acc,
        'f1_macro': f1,
    }

def main():
    df = pd.read_csv("data\\amazon_review_comments.csv")
    df = df.dropna(subset=['cleaned_review'])
    df['cleaned_review'] = df['cleaned_review'].astype(str)
    df['sentiments'] = df['sentiments'].str.lower()

    df = balancear_dataset(df, 'sentiments')
    y = df['sentiments'].values
    resultados = []

    # ===== spaCy =====
    print("\n🔹 Vetorizando com spaCy...")
    nlp_spacy = spacy.load("en_core_web_lg", disable=["tagger", "parser", "ner"])
    docs = list(tqdm(nlp_spacy.pipe(df['cleaned_review'], batch_size=64), total=len(df)))
    X_spacy = np.vstack([doc.vector for doc in docs])
    resultados.append(avaliar_modelo(X_spacy, y, "spaCy"))

    # ===== BERT =====
    print("\n🔸 Vetorizando com BERT...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando device: {device}")

    bert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    cache_file = "embeddings_bert.npy"
    if os.path.exists(cache_file):
        print("Carregando embeddings BERT do cache...")
        X_bert = np.load(cache_file)
    else:
        print("Gerando embeddings BERT (isso pode levar alguns minutos na primeira vez)...")
        X_bert = bert_model.encode(
            df['cleaned_review'].tolist(),
            batch_size=128,
            show_progress_bar=True,
            device=device,
            convert_to_numpy=True
        )
        np.save(cache_file, X_bert)
    resultados.append(avaliar_modelo(X_bert, y, "BERT"))

    # ===== Comparação de performance geral =====
    df_result = pd.DataFrame(resultados)
    df_result.set_index('model')[['accuracy', 'f1_macro']].plot.bar(
        rot=0, figsize=(8, 5), title="Comparação Geral dos Modelos"
    )
    plt.ylabel("Pontuação")
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

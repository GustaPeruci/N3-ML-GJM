import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
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

def main():
    df = pd.read_csv("data\\amazon_review_comments.csv")

    df = df.dropna(subset=['cleaned_review'])
    df['cleaned_review'] = df['cleaned_review'].astype(str)
    df['sentiments'] = df['sentiments'].str.lower()

    df = balancear_dataset(df, 'sentiments')
    y = df['sentiments'].values

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando device: {device}")

    bert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    cache_file = "embeddings_bert.npy"
    if os.path.exists(cache_file):
        print("Carregando embeddings BERT do cache...")
        X_bert = np.load(cache_file)
    else:
        print("Gerando embeddings BERT (pode demorar na primeira vez)...")
        X_bert = bert_model.encode(
            df['cleaned_review'].tolist(),
            batch_size=128,
            show_progress_bar=True,
            device=device,
            convert_to_numpy=True
        )
        np.save(cache_file, X_bert)

    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X_bert, y, test_size=0.2, stratify=y, random_state=42
    )

    # Treinar modelo
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Avaliar
    y_pred = model.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # Testar frases
    def testar_frase(frase):
        vec = bert_model.encode([frase], device=device, convert_to_numpy=True)
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        print(f'\nFrase: "{frase}"')
        print(f'Sentimento previsto: {pred}')
        print('Probabilidades:')
        for classe, p in zip(model.classes_, proba):
            print(f'  {classe}: {p:.4f}')

    testar_frase("This product is more less.")

if __name__ == "__main__":
    main()

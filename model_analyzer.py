import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def balancear_dataset(df, coluna_classe):
    contagens = df[coluna_classe].value_counts()
    max_count = contagens.max()
    print(f"Balanceando para {max_count} instâncias por classe")

    # Oversample para cada classe
    balanced_df = pd.concat([
        df[df[coluna_classe] == classe].sample(n=max_count, replace=True, random_state=42)
        for classe in contagens.index
    ])
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # embaralhar


def main():
    # Carregar o modelo spaCy com apenas a parte de vetores
    nlp = spacy.load("en_core_web_lg", disable=["tagger", "parser", "ner"])

    # Habilitar barra de progresso para nlp.pipe
    tqdm.pandas()

    # 1. Ler o dataset
    df = pd.read_csv("data\\amazon_review_comments.csv")

    # 2. Remover nulos e forçar string
    df = df.dropna(subset=['cleaned_review'])
    df['cleaned_review'] = df['cleaned_review'].astype(str)
    df['sentiments'] = df['sentiments'].str.lower()

    # 3. Balancear as classes com undersampling
    df = balancear_dataset(df, 'sentiments')

    # 4. Vetorização otimizada com spaCy e pipe
    print("Gerando vetores com spaCy (pode levar alguns segundos)...")
    docs = list(tqdm(nlp.pipe(df['cleaned_review'], batch_size=64), total=len(df)))
    X_vec = np.vstack([doc.vector for doc in docs])
    y = df['sentiments'].values

    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, stratify=y, random_state=42
    )

    # 6. Treinamento
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 7. Avaliação
    y_pred = model.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # 8. Testar frases
    def testar_frase(frase):
        vec = nlp(frase).vector.reshape(1, -1)
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        print(f'\nFrase: "{frase}"')
        print(f'Sentimento previsto: {pred}')
        print('Probabilidades:')
        for classe, p in zip(model.classes_, proba):
            print(f'  {classe}: {p:.4f}')

    testar_frase("this product don't work at all, want my money back")

if __name__ == "__main__":
    main()

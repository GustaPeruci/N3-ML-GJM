import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # Carregar o modelo spaCy
    nlp = spacy.load("en_core_web_lg")

    # Habilitar barra de progresso para nlp.pipe
    tqdm.pandas()

    # 1. Ler o dataset
    df = pd.read_csv("data\\amazon_review_comments.csv")

    # 2. Remover nulos e forçar string
    df = df.dropna(subset=['cleaned_review'])
    df['cleaned_review'] = df['cleaned_review'].astype(str)
    df['sentiments'] = df['sentiments'].str.lower()

    # 3. Vetorização eficiente com tqdm + nlp.pipe
    print("Gerando vetores com spaCy (pode levar alguns segundos)...")
    docs = list(tqdm(nlp.pipe(df['cleaned_review'], batch_size=64, n_process=-1), total=len(df)))
    X_vec = np.vstack([doc.vector for doc in docs])
    y = df['sentiments'].values

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, stratify=y, random_state=42
    )

    # 5. Treinamento
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 6. Avaliação
    y_pred = model.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # 7. Testar frases
    def testar_frase(frase):
        vec = nlp(frase).vector.reshape(1, -1)
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        print(f'\nFrase: "{frase}"')
        print(f'Sentimento previsto: {pred}')
        print('Probabilidades:')
        for classe, p in zip(model.classes_, proba):
            print(f'  {classe}: {p:.4f}')

    testar_frase("It does the job, although there are some minor things that could be improved.")

if __name__ == "__main__":
    main()

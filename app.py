import streamlit as st
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from scipy.spatial import distance
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from joblib import load
import faiss

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

films = pd.read_csv('movies_2.csv').dropna()
films['description'] = films['description'].astype(str)

def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=1024)
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()

embeded_list = load('embeded_list.joblib')
index = faiss.IndexFlatL2(embeded_list.shape[1])
index.add(embeded_list.astype('float32'))

text = st.text_input('Введите текст')
count_visible = st.number_input("Введите количество отображаемых элементов", 1, 10, 5, step=1)
if st.button("Найти", type="primary"):
    st.write('Количество фильмов в выборке 4950')
    if text and count_visible:
        embeded_text = embed_bert_cls(text, model, tokenizer).reshape(1,-1)
        D, I = index.search(embeded_text, index.ntotal)
        # cossim = pairwise_distances(embeded_text, embeded_list)[0]
        for i in range(count_visible):
            col1, col2 = st.columns(2)
            with col1:
                st.header(films.iloc[I[0]].iloc[i][2])
                st.write(films.iloc[I[0]].iloc[i][3].replace('\xa0', ' '))
                st.write(f'Мера схожести евклидова расстояния {D[0][i]:4f}')
            with col2:
                try:
                    st.image(films.iloc[I[0]].iloc[i][1])
                except:
                    st.write('Нет картинки')
        st.header('Самый не подходящий запрос')
        col3, col4 = st.columns(2)
        with col3:
            st.header(films.iloc[I[0]].iloc[-1][2])
            st.write(films.iloc[I[0]].iloc[-1][3].replace('\xa0', ' '))
            st.write(f'Мера схожести евклидова расстояния {D[0][i]:.4f}')
        with col4:
            st.image(films.iloc[I[0]].iloc[-1][1])
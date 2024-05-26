import telebot
from sklearn.metrics.pairwise import pairwise_distances
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import torch
from joblib import load
import faiss

# Initialize Telegram Bot
bot = telebot.TeleBot(token='7051493258:AAGrCa8TjijtEufPYZWzqdVGN8mN3gQQ-WU')

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

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, "Привет! Я бот для поиска фильмов 🤖. Отправь мне текстовый запрос, и я постараюсь помочь. Чтобы поиск был удачный, напиши подробный запрос 🆘")

@bot.message_handler(func=lambda message: True)
def search_films(message):
    text = message.text
    embeded_text = embed_bert_cls(text, model, tokenizer).reshape(1,-1)
    D, I = index.search(embeded_text, index.ntotal)
    count_visible = 3  # Задайте количество отображаемых элементов
    for i in range(count_visible):
        movie_title = films.iloc[I[0]].iloc[i]['movie_title']
        description = films.iloc[I[0]].iloc[i]['description'].replace('\xa0', ' ')
        movie_url = films.iloc[I[0]].iloc[i]['movie_url']
        image_url = films.iloc[I[0]].iloc[i]['image_url']
        film_message = f"<b>{movie_title}</b>\n{description}\nMovie URL: {movie_url}"
        bot.send_photo(message.chat.id, image_url, caption=film_message, parse_mode='HTML')

bot.polling()

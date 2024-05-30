
import pandas as pd
import nltk
import numpy as np
import telebot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import textblob
import re
from app.chat import chatbot
from flask import Flask, render_template, request, jsonify
from os import getenv
from app.auth_t import token
from app.db_call import QASystem
token = token
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
df = pd.read_csv("data.csv", encoding='unicode_escape')
questions_list = df['Questions'].tolist()
answers_list = df['Answers'].tolist()

def Bot(token):
    bot = telebot.TeleBot(token)
    qa = QASystem('data.csv')
    @bot.message_handler(commands=["start"])
    def start_message(message):
        bot.send_message(message.chat.id, "Здравствуйте, отправьте любое сообщение для старта ")
    @bot.message_handler(func=lambda message: True)
    def handle_message(message):
        response = chatbot(message)
        bot.send_message(message.chat.id, response)
    def ask():
        parser = parser()
        message = str(request.form['messageText'])
        corrected_text = parser(message)
        print(corrected_text['result'])
        bot_response = chatbot(corrected_text['result']) 
        return jsonify({'status':'OK','answer':bot_response})

            
    bot.polling()
if __name__ == '__main__':
    Bot(token)
    chatbot()
    

def parser(text):
    blob = textblob.TextBlob(text)
    corrected_text = blob.correct()
    return corrected_text


def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    text = re.sub(r'[^\w\s]', '', text)  # Remove non-alphanumeric characters
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
    return ' '.join(stemmed_tokens)

def preprocess_with_stopwords(text):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    text = re.sub(r'[^\w\s]', '', text)  # Remove non-alphanumeric characters
    tokens = nltk.word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
    return ' '.join(stemmed_tokens)

vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
X = vectorizer.fit_transform([preprocess(q) for q in questions_list])

def get_response(text):
    processed_text = preprocess_with_stopwords(text)
    print("processed_text:", processed_text)
    vectorized_text = vectorizer.transform([processed_text])
    similarities = cosine_similarity(vectorized_text, X)
    print("similarities:", similarities)
    max_similarity = np.max(similarities)
    print("max_similarity:", max_similarity)
    if max_similarity > 0.6:
        high_similarity_questions = [q for q, s in zip(questions_list, similarities[0]) if s > 0.6]
        print("high_similarity_questions:", high_similarity_questions)

        target_answers = []
        for q in high_similarity_questions:
            q_index = questions_list.index(q)
            target_answers.append(answers_list[q_index])
        print(target_answers)

        Z = vectorizer.fit_transform([preprocess_with_stopwords(q) for q in high_similarity_questions])
        processed_text_with_stopwords = preprocess_with_stopwords(text)
        print("processed_text_with_stopwords:", processed_text_with_stopwords)
        vectorized_text_with_stopwords = vectorizer.transform([processed_text_with_stopwords])
        final_similarities = cosine_similarity(vectorized_text_with_stopwords, Z)
        closest = np.argmax(final_similarities)
        return target_answers[closest]
    else:
        return "Не могу ответить на этот вопрос"
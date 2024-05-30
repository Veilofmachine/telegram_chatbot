from db_call import QASystem
import pandas as pd
import csv

def chatbot(message):

    qa = QASystem('data.csv')
    qa_response = None
    try:
        qa_response = qa.get_response(message.text)
        print("Try")
    except:
        qa_response = "Не могу ответить на этот вопрос."
        print("Except, in qa_response")

    return qa_response
from email import message
from flask import Flask, redirect, url_for, request, render_template
import sys
import requests
from flask import Response
import json
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import svm
import pandas as pd
from googletrans import Translator
from transliteration import *
import pickle

app = Flask(__name__, template_folder='Templates')
context_set = ""


def write_json(data, filename='response.json'):
    with open(filename, 'w') as f:
        json.dump(data, f, intent=4, ensure_asci=False)


def getFeatures(input_words):

    model = pickle.load(open('./Language_Detection_Model/model.pkl', 'rb'))
    feature_list = []

    features_en_suffixes = ['eer', 'ion', 'ism', 'ity', 'or', 'sion', 'ship', 'th', 'able', 'ible', 'al',
                            'ant', 'ary', 'ful', 'ic', 'ious', 'ous', 'y', 'ed', 'en', 'er', 'ing', 'ise', 'ly', 'ward', 'wise']
    features_en_prefixes = ['anti', 'auto', 'circum', 'dis', 'ex', 'extra', 'hetero', 'homo', 'hyper', 'em', 'ir', 'inter', 'intra',
                            'macro', 'mono', 'non', 'omni', 'post', 'pre', 'pro', 'sub', 'sym', 'syn', 'tele', 'trans', 'tri', 'uni', 'up', 'tele']
    features_sinhala_letters = ['අ',	'ආ',	'ඇ',	'ඈ',	'ඉ',	'ඊ',	'උ',	'ඌ',	'එ', 'ඒ',	'ඔ',	'ඕ',	'ක',	'ච', 'ට',	'ත',	'ප',	'ද',	'ග',	'ජ',	'ඩ',	'බ',	'ම',	'ය',	'ර',	'ල',	'ව',	'ශ',	'ෂ',	'ස',	'හ',
                                'ළ',	'ෆ',	'ඛ',	'ඡ',	'ඨ',	'ථ',	'ඝ',	'ඵ',	'ධ',	'ඣ',	'භ',	'a',	'b',	'c',	'd',	'e',	'f',	'g',	'h',	'i',	'j',	'k',	'l',	'm',	'n',	'o',	'p',	'q',	'r',	's',	't',	'u',	'v',	'w',	'x',	'y',	'z']

    for word in input_words:
        feature_list.clear()
        for item in features_en_suffixes:
            if word.endswith(item):
                feature_list.append(1)
            else:
                feature_list.append(0)
        for item in features_en_prefixes:
            if word.startswith(item):
                feature_list.append(1)
            else:
                feature_list.append(0)
        for item in features_sinhala_letters:
            if item in word:
                feature_list.append(1)
            else:
                feature_list.append(0)
        if word.endswith('a'):
            feature_list.append(1)
        else:
            feature_list.append(0)
        print('For loop End')
        print(len(feature_list))
        print(feature_list)

    return feature_list


@app.route('/', methods=['POST', 'GET'])
def index():

    sinhala_list = []
    english_list = []
    singlish_list = []
    translator = Translator()
    text = ''

    first_dominant_lng = ''

    if request.method == 'POST':

        msg = request.get_json()
        print(msg, file=sys.stdout)
        model = pickle.load(open('model.pkl', 'rb'))
        input_words = msg['message']['text'].split()
        print(input_words, file=sys.stdout)
        print("dic_list:")
        print(dic_list)
        print("dic_list:")

        for word in input_words:
            feature_list = getFeatures(word)
            print('Predicted'+model.predict([feature_list]), file=sys.stdout)

            if model.predict([feature_list]) == 'Singlish':
                transliterated_word = transliterate(word)
                sinhala_list.append(transliterated_word)
                print(sinhala_list)
            elif model.predict([feature_list]) == 'Sinhala':
                sinhala_list.append(word.encode('utf-8').decode('utf-8'))
            else:
                english_list.append(word)

       
        print(sinhala_list)
        print(english_list)
        print(singlish_list)

        if(len(singlish_list) >= len(english_list)):
            if(len(singlish_list) >= len(sinhala_list)):
                highest = 'Singlish'
            else:
                highest = 'Sinhala'

        else:
            if(len(english_list) >= len(sinhala_list)):
                highest = 'English'
            else:
                highest = 'Sinhala'

        if highest == 'Sinhala':
            for word in english_list:
                result = translator.translate(word, src='en', dest='si')
                sinhala_list.append(result.text)
             
            for item in sinhala_list:
                text = str(text+" "+item)

           
            data = json.dumps(
                {"sender": msg["message"]["from"]["id"], "message": text})
            print(data)
            headers = {'Content-type': 'application/json',
                       'Accept': 'text/plain'}
            res = requests.post(
                'https://3256-123-231-122-161.in.ngrok.io/webhooks/rest/webhook', data=data, headers=headers)
            res = res.json()
            print(res, file=sys.stdout)

            for e in range(len(res)):
                id_to_telegram = msg["message"]["from"]["id"]
                message_to_telegram = res[int(e)]["text"]
                data_to_telegram = json.dumps(
                    {"chat_id": id_to_telegram, "text": message_to_telegram})
                print(data_to_telegram)
                res_from_telegram = requests.post(
                    'https://api.telegram.org/bot5155776387:AAFoUElwB4BgZHeqS4sKNcXpfcfDyHbAq2A/sendMessage', data=data_to_telegram, headers=headers)

        if highest == 'English':
            for word in sinhala_list:
                result = translator.translate(word, src='si', dest='en')
                english_list.append(result.text)
               
            for item in english_list:
                text = str(text+" "+item)

    
            data = json.dumps(
                {"sender": msg["message"]["from"]["id"], "message": text})
            print(data)
            headers = {'Content-type': 'application/json',
                       'Accept': 'text/plain'}
            res = requests.post(
                'https://6879-123-231-122-161.ngrok.io/webhooks/rest/webhook', data=data, headers=headers)
            res = res.json()

            print(res, file=sys.stdout)

            for e in range(len(res)):
                id_to_telegram = msg["message"]["from"]["id"]
                message_to_telegram = res[int(e)]["text"]
                data_to_telegram = json.dumps(
                    {"chat_id": id_to_telegram, "text": message_to_telegram})
                res_from_telegram = requests.post(
                    'https://api.telegram.org/bot5155776387:AAFoUElwB4BgZHeqS4sKNcXpfcfDyHbAq2A/sendMessage',  data=data_to_telegram, headers=headers)
                print(res_from_telegram, file=sys.stdout)

        return Response('Ok', status=200)

    else:
        return '<p>Covid Chatbot!!</p>'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=50000, debug=True)

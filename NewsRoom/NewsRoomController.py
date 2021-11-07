from flask import Flask,render_template
from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import pandas as pd
import numpy as np
import json
from tqdm import tqdm, trange
import torch
from tensorflow import keras
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
#from tensorflow_core import keras
import os
from nltk.tokenize import sent_tokenize
from keybert import KeyBERT
import requests

app = Flask(__name__)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
bert_from_pickle = pd.read_pickle(r'models/custom-T5')
keybert_from_pickle = KeyBERT('distilbert-base-nli-mean-tokens')
api_url="https://customsearch.googleapis.com/customsearch/v1?key=AIzaSyBPXEErAKKCOVt3CLve1FqhykpsmfIWrHU&cx=26416585bbd9da8fa&q={query}&searchType=image"
    #torch.load('models/key-bert',map_location =torch.device('cpu'))

@app.route('/newsroom')
def hello():
    return 'Hie I am working fine'


def getSummarisedText(data):
    text=data["text"]
    input_ids = tokenizer.encode("summarise:"+text, return_tensors='pt')
    # Generating summary ids
    summary_ids = bert_from_pickle.generate(
        input_ids=input_ids,

        max_length=150,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )
    # Decoding the tensor and printing the summary.
    t5_summary = tokenizer.decode(summary_ids[0])
    return t5_summary.replace("<pad>","").replace("<extra_id_0>","")


def getKeyWords(data):
    text = data["text"]
    tkns = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tkns)
    segments_ids = [0] * len(tkns)
    tokens_tensor = torch.tensor([indexed_tokens]).to("cpu")
    segments_tensors = torch.tensor([segments_ids]).to("cpu")

    prediction = []
    logit = keybert_from_pickle.extract_keywords(text, stop_words='english', use_mmr=True, diversity=0.5,top_n=10)
    for item in logit:
        if not item[0].isdigit():
          prediction.append(item[0])



    return json.dumps(prediction)


@app.route('/newsroom/summarise',methods=["GET","POST"])
def getSummary():
    data = request.json
    return getSummarisedText(data)


@app.route('/newsroom/getkeywords',methods=["GET","POST"])
def getKeywords():
    data = request.json
    return getKeyWords(data)

@app.route('/newsroom/getImages',methods=["GET","POST"])
def getImages():
    data = request.json
    keywords=data["keywords"]
    url=api_url.replace("{query}",keywords)
    result=json.loads(requests.get(url).text)
    print(str(result))
    listi=[]
    items=result["items"]
    for i in items:
        print(i)
        value=i['image']['thumbnailLink']+"####"+i['htmlTitle']
        listi.append(value)
    return json.dumps(listi)



@app.route('/renderTemplate')
def html():
    return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)
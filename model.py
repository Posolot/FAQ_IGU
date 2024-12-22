import json
import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
tf.__version__
import os
import shutil
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
import matplotlib.pyplot as plt
import numpy as np


with open("faq_data.json", "r", encoding="utf-8") as file:
    faq_data = json.load(file)

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

tf.get_logger().setLevel('ERROR')

bert = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1"
bpre = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

bert_preprocess_model = hub.KerasLayer(bpre)
text_test = ['this is such and amazing movie!']
text_preprocessed = bert_preprocess_model(text_test)

bert_model = hub.KerasLayer(bert)
bert_results = bert_model(text_preprocessed)

embedding = bert_results["pooled_output"].numpy()


dataset = []
for i in range(len(questions)):
    q_emb = bert_model(bert_preprocess_model(questions[i:i+1]))["pooled_output"].numpy()#.sum(axis=1)
    a_emb = bert_model(bert_preprocess_model(answers[i:i+1]))["pooled_output"].numpy()#.sum(axis=1)
    dataset.append([np.array(q_emb[0]),np.array(a_emb[0])])
dataset = np.array(dataset)
np.save('dataset1', dataset)

X, Y = [], []
for i in range(dataset.shape[0]):
    for j in range(dataset.shape[0]):
        X.append(np.concatenate([dataset[i,0,:],dataset[j,1,:]],axis=0))
        if i == j:
            Y.append(1)
        else:
            Y.append(0)
X = np.array(X)
Y = np.array(Y)

early_stopping = EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True, mode='max')
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(1024,)))
model.add(tf.keras.layers.Dense(100,activation='selu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
es=tf.keras.callbacks.EarlyStopping(monitor='auc',mode='max',patience=10,restore_best_weights=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC(curve='pr',name='auc')])
model.fit(X,Y,epochs=1000,class_weight={0:1,1:8}, callbacks=[early_stopping], validation_split=0.2)
model.summary()

model.save("model1.keras")

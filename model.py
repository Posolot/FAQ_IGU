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


with open("faq_data.json", "r", encoding="utf-8") as file:
    faq_data = json.load(file)

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]
questions=list(map(lambda x: x.lower(),questions))
answers=list(map(lambda x: x.lower(),answers))


tf.get_logger().setLevel('ERROR')

bert="https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3"
bpre="https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3"

bert_preprocess_model=hub.KerasLayer(bpre)
bert_model = hub.KerasLayer(bert)

dataset=[]
questions=list(map(lambda x: x.lower(),questions))
answers=list(map(lambda x: x.lower(),answers))
for i in range(len(questions)):
  q_emb=bert_model(bert_preprocess_model(questions[i:i+1]))["pooled_output"].numpy()#.sum(axis=1)
  a_emb=bert_model(bert_preprocess_model(answers[i:i+1]))["pooled_output"].numpy()#.sum(axis=1)
  dataset.append([np.array(q_emb[0]),np.array(a_emb[0])])
dataset=np.array(dataset)


np.save('dataset1', dataset)

X,Y=[],[]
for i in range(dataset.shape[0]):
  for j in range(dataset.shape[0]):
    X.append(np.concatenate([dataset[i,0,:],dataset[j,1,:]],axis=0))
    if i==j:
      Y.append(1)
    else:
      Y.append(0)
X=np.array(X)
Y=np.array(Y)
X.shape,Y.shape,Y

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(1536,)))
model.add(tf.keras.layers.Dense(200,activation="relu"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss="binary_crossentropy",metrics=[tf.keras.metrics.AUC(curve="pr",name="auc")])
model.fit(X,Y,epochs=1000)
model.summary()

model.save("model1.keras")

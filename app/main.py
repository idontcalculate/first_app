#!/usr/bin/env python
# coding: utf-8
from fastapi import FastAPI
from fastapi.responces import StreamingResponse

app=FastAPI():
@app.get("/predict")
def get_y_predict():
    import tensorflow as tf
    import tensorflow_hub as hub
    import pandas as pd

    df= pd.read_csv('SMSSpamCollection.csv', sep='\t',
                           names=["label", "message"], encoding='utf-8')
    df.head()

    df.rename(columns = {'label':'Category', 'message':'Message'}, inplace = True)
    df.head()

    df.groupby('Category').describe()

    df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)
    df.head()




    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(df['Message'],df['spam'], stratify=df['spam'])

    X_train.head(4)



    get_ipython().system('pip install tensorflow-text')




    get_ipython().system('pip install tensorflow-text')




    import tensorflow_text as text




    bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")



    def get_sentence_embeding(sentences):
        preprocessed_text = bert_preprocess(sentences)
        return bert_encoder(preprocessed_text)['pooled_output']

    get_sentence_embeding([
    "500$ discount. hurry up", 
    "Jim, are you up for a volleybal game tomorrow?"]
    )





# Bert layers
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)

# Neural network layers
    model = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
    model = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(model)

# Use inputs and outputs to construct a final model
    model = tf.keras.Model(inputs=[text_input], outputs = [model])





    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])





    model.fit(X_train, y_train, epochs=3, batch_size = 32)





    model.evaluate(X_test, y_test)


    y_predicted = model.predict(X_test)
    y_predicted = y_predicted.flatten()

    return print(y_predicted)








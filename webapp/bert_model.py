from transformers import BertTokenizer, TFBertModel
import joblib
from tensorflow import keras
import tensorflow as tf
import gdown
import os
import zipfile
from arabert.preprocess import ArabertPreprocessor

# Load the saved model


class BERTModel():
    def __init__(self) -> None:
        # self.model = keras.models.load_model('../models/dlmodels/arabertv5_0')

        self.max_length_tokens = 64
        self.bert_model_name = 'aubmindlab/bert-base-arabertv02-twitter'
        self.tokenizer = BertTokenizer.from_pretrained(
        self.bert_model_name, max_length=self.max_length_tokens, model_max_length=self.max_length_tokens)

        self.prep_model_name = 'aubmindlab/bert-base-arabertv2'

        self.arabert_prep = ArabertPreprocessor(model_name=self.prep_model_name)
        self.directory = 'model5bert/arabertv5_0'
        self.model_loaded = False
        
        self.load_gd()
        self.model = keras.models.load_model(
            "model5bert/arabertv5_0/", custom_objects={"TFBertModel": TFBertModel})


    def load_gd(self):

        f_checkpoint = "model5bert"
        gd_link = "https://drive.google.com/file/d/1XmNE4Vl4kxFA_C9fjL5IB7QCzDg_xNhL/view?usp=sharing"


        if not os.path.exists(f_checkpoint):
            print("downloading model")
            res = gdown.download(gd_link, f_checkpoint, quiet=True,
                                fuzzy=True, use_cookies=False)
            unzip_file("bertmodel5.zip", "model5bert")
        else:
            print("model already downloaded")
        self.model_loaded = True


    def predict(self, txt):
        if self.model_loaded is False:
            return {'status':'model not loaded yet'}
        x = self.preprocess(txt)
        res = self.model.predict(x)
        return res

    def preprocess(self, txt):
        txt_res = self.arabert_prep.preprocess(txt)
        tokens = self.tokenizer(txt_res, padding='max_length',
                        truncation=True, return_tensors='tf', max_length=self.max_length_tokens).input_ids
        x = tf.convert_to_tensor(tokens)
        return x


def unzip_file(zip_filename, output_folder):
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(output_folder)

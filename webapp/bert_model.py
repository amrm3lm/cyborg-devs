import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import joblib
from tensorflow import keras

# Load the saved model


class BERTModel():
    def __init__(self) -> None:
        # self.model = keras.models.load_model('../models/dlmodels/arabertv5_0')

        self.max_length_tokens = 64
        self.bert_model_name = 'aubmindlab/bert-base-arabertv02-twitter'
        self.tokenizer = BertTokenizer.from_pretrained(
        self.bert_model_name, max_length=self.max_length_tokens, model_max_length=mself.ax_length_tokens)

        self.prep_model_name = 'aubmindlab/bert-base-arabertv2'
        self.arabert_prep = ArabertPreprocessor(model_name=self.prep_model_name)

    def predict(self, txt):
        x = self.preprocess(txt)
        res = self.model.bert.predict(x)
        return res

    def preprocess(self, txt):
        txt_res = self.arabert_prep.preprocess(txt)
        tokens = self.tokenizer(txt_res, padding='max_length',
                        truncation=True, return_tensors='tf', max_length=self.max_length_tokens).input_ids
        x = tf.convert_to_tensor(tokens)
        return x

from arabert.preprocess import ArabertPreprocessor
import joblib
import re
import nltk
import scipy.sparse as sp
from ruqiya import ruqiya

emoj = re.compile("["
                  u"\U00002700-\U000027BF"  # Dingbats
                  u"\U0001F600-\U0001F64F"  # Emoticons
                  u"\U00002600-\U000026FF"  # Miscellaneous Symbols
                  u"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols And Pictographs
                  u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                  u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                  u"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                  u"\U0001F600-\U0001F64F"  # emoticons
                  u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                  u"\U0001F680-\U0001F6FF"  # transport & map symbols
                  u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                  u"\U00002702-\U000027B0"
                  u"\U000024C2-\U0001F251"
                  "]+", re.UNICODE)

bert_model_name = 'aubmindlab/bert-base-arabertv2'
# arabert_prep = ArabertPreprocessor(
#     model_name=bert_model_name, apply_farasa_segmentation=False)
tfidfVectorizer = joblib.load('webapp/tfidf_vectorizer.pkl')
scaler = joblib.load('webapp/scaler.pkl')
INPUT_LENTH = 19095

class SVMModel():
    def __init__(self) -> None:
        self.model = joblib.load('webapp/SVmModel.pkl')
        
    def predict(self, txt):
        X_combined = prepare_input(txt)
        res = self.model.predict(X_combined)
        print("RES-------------------------", res)
        return res


class LRModel():
    def __init__(self) -> None:
        self.model = joblib.load('webapp/lrModel.pkl')

    def predict(self, txt):
        X_combined = prepare_input(txt)
        res = self.model.predict(X_combined)
        print("RES-------------------------",res)
        return res
    
# todo : make the same scikit version
# class DTModel():
#     def __init__(self) -> None:
#         self.model = joblib.load('../Full/dt_model.pkl')

#     def predict(self, txt):
#         X_combined = prepare_input(txt)
#         res = self.model.predict(X_combined)
#         return res



        
def prepare_input(txt):
    features = preprocess_txt(txt)
    idfed = tfidfVectorizer.transform([features['puretext']])

    other_features = [[features['num_hashtags'],
                       features['num_mentions'], features['num_words'], features['num_emojis'], features['num_links'], features['num_emojis'], 0]]
    tformed = scaler.transform(other_features)
    X_combined = sp.hstack((idfed, tformed), format='csr')
    return X_combined

        

def preprocess_txt(txt):
    # feature extraction 
    features = feature_extract(txt)
    
    # cleaning
    txt = ruqiya.clean_text(txt)
    txt = remove_english_letters(txt)
    
    #tokeniz
    tokens = nltk.word_tokenize(txt)
    
    #stem
    st = nltk.ISRIStemmer()
    stemming_root = [st.stem(token) for token in tokens]
    stemmed = ' '.join(stemming_root)
    
    res = {'clean_text':txt, 'puretext':stemmed, 'tokenized':tokens, **features}
    return res

def remove_english_letters(text):
    pattern = re.compile(r"[a-zA-Z]")
    return re.sub(pattern, "", text)


def feature_extract(txt):
    res = {}
    res['num_emojis'] =  len(re.findall(emoj, txt))
    res['num_hashtags'] = len(re.findall(r'#\w+', txt))
    res['num_mentions'] = len(re.findall(r'@\w+', txt))
    res['num_words'] = len(txt.split())
    res['num_links'] = len(re.findall(r"(?:http?\://|https?\://|www)\S+", txt))
    
    return res

#import Libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# veri setinin içeri aktarılması
df = pd.read_csv("IMDB_Dataset.csv")

# metin verilerinin alınması

documents = df["review"]
labels = df["sentiment"] #positive veya negatif

# metin temizleme
def clean_text(text):
    
    #buyuk kucuk harf
    text = text.lower()
    
    #rakamları temizleme
    text = re.sub(r"\d+", "", text)
    
    #ozel karakterlerin temzilenmesi
    text = re.sub(r"[^\w\s]", "", text)
    
    #kısa kelimelerin temizlenmesi
    text = " ".join(word for word in text.split() if len(word)>2)
    
    #stop words silinmesi
    text = " ".join(word for word in text.split() if word.lower() not in stop_words)
    
    return text #temizlenmiş text'i return et
# metinleri temzileme
cleaned_doc = [clean_text(row) for row in documents]
# %% bow
#vectorizer tanımlama
vectorizer = CountVectorizer()

#metin -> sayısal hale getirme
X = vectorizer.fit_transform(cleaned_doc[:100]) 

#kelime kümesi gösterme
feature_names = vectorizer.get_feature_names_out()

#vektör temsili gösterme
vector_temsili = X.toarray()

df_bow = pd.DataFrame(vector_temsili, columns = feature_names)

#kelime frekansları gösterme
word_counts = X.sum(axis=0).A1
word_freg = dict(zip(feature_names, word_counts))

# İlk 5 keilmeyi print etme
most_common_5_word = Counter(word_freg).most_common(5)
print(f"most_common_5_words : {most_common_5_word}")





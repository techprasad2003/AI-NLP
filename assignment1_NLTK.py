import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

sample_text = """
Hello this is a sample text !. Ok !
"""

tokens = word_tokenize(sample_text)
print(" Tokens:")
print(tokens)


tokens_no_punct = [word for word in tokens if word not in string.punctuation]
print("\n After Punctuation Removal:")
print(tokens_no_punct)


stop_words = set(stopwords.words('english'))
tokens_no_stopwords = [word for word in tokens_no_punct if word.lower() not in stop_words]
print("\n After Stop Word Removal:")
print(tokens_no_stopwords)


porter = PorterStemmer()
porter_stemmed = [porter.stem(word) for word in tokens_no_stopwords]
print("\n Porter Stemmer:")
print(porter_stemmed)


lancaster = LancasterStemmer()
lancaster_stemmed = [lancaster.stem(word) for word in tokens_no_stopwords]
print("\n Lancaster Stemmer:")
print(lancaster_stemmed)


snowball = SnowballStemmer("english")
snowball_stemmed = [snowball.stem(word) for word in tokens_no_stopwords]
print("\n Snowball Stemmer:")
print(snowball_stemmed)


lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in tokens_no_stopwords]
print("\n Lemmatized Words:")
print(lemmatized_words)
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string


nlp = spacy.load("en_core_web_sm")


sample_text = """
Hello this is a sample text !. Ok !
"""


doc = nlp(sample_text)


tokens = [token.text for token in doc]
print("\n Tokens:")
print(tokens)


tokens_no_punct = [token.text for token in doc if token.text not in string.punctuation]
print("\n After Punctuation Removal:")
print(tokens_no_punct)


tokens_no_stopwords = [token.text for token in doc if not token.is_stop and token.text not in string.punctuation]
print("\n  After Stop Word Removal:")
print(tokens_no_stopwords)


lemmatized = [token.lemma_ for token in doc if not token.is_stop and token.text not in string.punctuation]
print("\n Lemmatized Words:")
print(lemmatized)


pos_tags = [(token.text, token.pos_) for token in doc]
print("\n POS Tags:")
print(pos_tags)

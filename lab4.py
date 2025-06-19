import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import spacy
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Ensure required resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Exercise 1: Modify the code to test different sentences
sentence = "Artificial intelligence is revolutionizing many industries."
tokens = word_tokenize(sentence)
pos_tags_nltk = pos_tag(tokens)
print("NLTK POS Tagging:", pos_tags_nltk)

# Using spaCy
print("spaCy POS Tagging:")
doc = nlp(sentence)
for token in doc:
    print(f"{token.text}: {token.pos_}")

# Exercise 2: Compare NLTK and spaCy taggers on a paragraph
paragraph = "Natural language processing is a crucial field in artificial intelligence. It enables machines to understand human language."
tokens_paragraph = word_tokenize(paragraph)
pos_tags_nltk_paragraph = pos_tag(tokens_paragraph)
print("\nNLTK POS Tagging on Paragraph:")
print(pos_tags_nltk_paragraph)

print("\nspaCy POS Tagging on Paragraph:")
doc_paragraph = nlp(paragraph)
for token in doc_paragraph:
    print(f"{token.text}: {token.pos_}")

# Exercise 3: Train a custom POS tagger with more data using Scikit-Learn
X_train = [{'word': 'The'}, {'word': 'quick'}, {'word': 'brown'}, {'word': 'fox'},
           {'word': 'jumps'}, {'word': 'over'}, {'word': 'the'}, {'word': 'lazy'}, {'word': 'dog'}]
y_train = ['DET', 'ADJ', 'ADJ', 'NOUN', 'VERB', 'ADP', 'DET', 'ADJ', 'NOUN']

vectorizer = DictVectorizer(sparse=False)
classifier = LogisticRegression(max_iter=200)
pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', classifier)
])

pipeline.fit(X_train, y_train)

# Predicting POS for a new word
X_test = [{'word': 'flies'}, {'word': 'run'}, {'word': 'blue'}]
print("\nCustom POS Tagger Predictions:")
print(pipeline.predict(X_test))

# Exercise 4: Explore deep learning-based POS tagging methods with Stanza
import stanza
stanza.download('en')
nlp_stanza = stanza.Pipeline('en')

doc_stanza = nlp_stanza(sentence)
print("\nStanza POS Tagging:")
for sentence in doc_stanza.sentences:
    for word in sentence.words:
        print(f"{word.text}: {word.xpos}")

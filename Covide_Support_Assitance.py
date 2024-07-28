import random
import string
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import download

from tkinter import *

# Download NLTK data
download('punkt', quiet=True)
download('wordnet', quiet=True)
download('stopwords', quiet=True)

# Load stopwords
stop_words = set(stopwords.words('english'))

# Reading in the JSON file
try:
    with open('covid.json', 'r', encoding='utf8') as json_file:
        intents = json.load(json_file)["intents"]
except FileNotFoundError:
    intents = []

# Tokenization and Preprocessing
def preprocess_text(text):
    """
    Preprocesses the input text by tokenizing, lemmatizing, and removing stopwords and punctuation.
    """
    tokens = word_tokenize(text)
    lemmer = WordNetLemmatizer()
    tokens = [lemmer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if token not in string.punctuation and token not in stop_words]
    return ' '.join(tokens)

# Initialize an empty corpus
corpus = []
corpus_data = []

# Load predefined data into the corpus
for intent in intents:
    for pattern in intent['patterns']:
        processed_pattern = preprocess_text(pattern)
        corpus.append(processed_pattern)
        corpus_data.append((processed_pattern, intent['tag']))

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# Save the corpus and TF-IDF vectorizer for incremental learning
def save_data():
    """
    Saves the current corpus to a JSON file for persistent storage.
    """
    with open('corpus.json', 'w') as f:
        json.dump(corpus, f)

# KeyWord matching and natural language
GREETING_INPUTS = (
    "hello", "hi", "greetings", "sup", "what's up", "hey", "good morning",
    "good afternoon", "good evening", "howdy", "hola", "yo", "hi there",
    "hey there", "hiya", "how are you", "how's it going"
)
GREETING_RESPONSES = [
    "Hi", "Hey", "Hello", "I am AI Bot! You are talking to me",
    "Greetings!", "Howdy!", "Hi there!", "Hey there!", "Hello! How can I assist you today?",
    "Hi! How are you?", "Hey! What's up?", "Hello! What can I do for you today?"
]

def greeting(sentence):
    """
    If user's input is a greeting, return a greeting response.
    """
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Generating response
def response(user_response):
    """
    Generates a response to the user input based on cosine similarity with the corpus.
    """
    if not corpus:
        return "I don't have enough information yet. Please provide more input."

    Assistance_Bot_response = ''
    user_response = preprocess_text(user_response)
    tfidf_user_response = vectorizer.transform([user_response])
    cosine_similarities = cosine_similarity(tfidf_user_response, vectorizer.transform(corpus))
    most_similar_idx = cosine_similarities.argmax()

    if cosine_similarities[0][most_similar_idx] == 0:
        return None
    else:
        tag = corpus_data[most_similar_idx][1]
        for intent in intents:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])

# Incremental learning: Append new data to the corpus and retrain
def learn_from_interaction(user_input, bot_response):
    """
    Learns from the interaction by appending new user input and bot response to the corpus and retraining the model.
    """
    global corpus, tfidf_matrix
    corpus.append(preprocess_text(user_input))
    corpus.append(preprocess_text(bot_response))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    save_data()

# GUI setup
root = Tk()
root.title('SupportAssist for Covid 19')
root.geometry('500x500')

# Create the chat area
chatWindow = Text(root, bd=1, bg='black', fg='white', width=50, height=8)
chatWindow.place(x=12, y=9, height=425, width=475)

# Create message area
messageWindow = Text(root, bg="white", width=30, height=4, font=('Arial', 16))
messageWindow.place(x=12, y=450, height=40, width=370)

# Create a button to send the message
send_button = Button(root, text='submit', bg='lime', activebackground='white', width=12, height=5, font=('Arial', 16), command=lambda: send_message())
send_button.place(x=387, y=450, height=40, width=100)

# Function to send message and get response
def send_message():
    """
    Handles the user input, generates a response, and updates the chat window.
    """
    user_message = messageWindow.get("1.0", END).strip()
    chatWindow.insert(END, "You: " + user_message + "\n")
    messageWindow.delete("1.0", END)
    
    if user_message.lower() == 'bye':
        chatWindow.insert(END, "Assistance Bot: Bye! Take care.\n")
        root.quit()
    elif user_message.lower() in ['thanks', 'thank you']:
        chatWindow.insert(END, "Assistance Bot: You are welcome..\n")
    elif user_message.lower() in ['how are you']:
        chatWindow.insert(END, "Assistance Bot: I'm fine, thank you..\n")
    elif user_message.lower() in ['good morning', 'morning']:
        chatWindow.insert(END, "Assistance Bot: Good morning.\n")
    elif user_message.lower() in ['good night']:
        chatWindow.insert(END, "Assistance Bot: Good night.\n")
    elif user_message.lower() in ['are you there']:
        chatWindow.insert(END, "Assistance Bot: Yes, I'm here, How can I assist you?\n")
    elif greeting(user_message) is not None:
        bot_response = greeting(user_message)
        chatWindow.insert(END, "Assistance Bot: " + bot_response + "\n")
    else:
        bot_response = response(user_message)
        
        if bot_response is None:
            chatWindow.insert(END, "Assistance Bot: I am sorry! I don't understand you. Can you please provide a suitable response?\n")
        else:
            chatWindow.insert(END, "Assistance Bot: " + bot_response + "\n")
            learn_from_interaction(user_message, bot_response)

root.mainloop()

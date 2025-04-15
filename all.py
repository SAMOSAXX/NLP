#COMMON IMPORTS
import numpy as np
import pandas as pd
import os
import re
# Disable WandB logging
os.environ["WANDB_DISABLED"] = "true"

#TEXT SUMMARIZATION
from datasets import Dataset
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments


#SENTIMENT ANALYSIS
from transformers import pipeline
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

#MACHINE TRANSLATION
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

#API
from openai import OpenAI
api_key = "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"
client = OpenAI(api_key=api_key)

# Get user input
user_message = input("Enter your message: ")

try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )
        print(response.choices[0].message.content)
except Exception as e:
        print(f"Error in semantic comparison: {e}")
        
        
#CYK
from collections import defaultdict
from nltk.tree import Tree
from tabulate import tabulate

# ---------------------------
# READ GRAMMAR FROM FILE
# Format: A -> B C OR A -> word
# ---------------------------
def read_grammar(filename):
    grammar = defaultdict(list)
    rhs_to_lhs = defaultdict(set)
    with open(filename) as f:
        for line in f:
            if '->' not in line:
                continue
            lhs, rhs = line.strip().split('->')
            lhs = lhs.strip()
            rhs_symbols = rhs.strip().split()
            grammar[lhs].append(rhs_symbols)
            rhs_to_lhs[tuple(rhs_symbols)].add(lhs)
    return grammar, rhs_to_lhs

# ---------------------------
# CYK PARSER WITH BACKPOINTERS
# ---------------------------
def cyk_parser(tokens, rhs_to_lhs):
    n = len(tokens)
    table = [[set() for _ in range(n)] for _ in range(n)]
    back = [[defaultdict(list) for _ in range(n)] for _ in range(n)]

    # Fill diagonals
    for i, token in enumerate(tokens):
        for lhs in rhs_to_lhs.get((token,), []):
            table[i][i].add(lhs)
            print(f"Matched terminal: {lhs}[1,{i+1}] -> {token}")

    # Fill upper triangle
    for l in range(2, n + 1):  # Span length
        for i in range(n - l + 1):  # Start index
            j = i + l - 1
            for k in range(i, j):
                for B in table[i][k]:
                    for C in table[k+1][j]:
                        for A in rhs_to_lhs.get((B, C), []):
                            table[i][j].add(A)
                            back[i][j][A].append((k, B, C))
                            print(f"Applied Rule: {A}[{l},{i+1}] --> {B}[{k-i+1},{i+1}] {C}[{j-k},{k+2}]")

    # Tree building
    def build_tree(i, j, symbol):
        if i == j:
            return (symbol, tokens[i])
        for k, B, C in back[i][j].get(symbol, []):
            left = build_tree(i, k, B)
            right = build_tree(k+1, j, C)
            return (symbol, left, right)

    return table, back, build_tree if 'S' in table[0][n-1] else None

# ---------------------------
# CONVERT TO nltk.Tree FOR DISPLAY
# ---------------------------
def tuple_to_nltk_tree(tree_tuple):
    if isinstance(tree_tuple, tuple):
        label = tree_tuple[0]
        children = [tuple_to_nltk_tree(child) for child in tree_tuple[1:]]
        return Tree(label, children)
    else:
        return tree_tuple

# ---------------------------
# PRINT CYK TABLE USING TABULATE
# ---------------------------
def print_table(table, tokens):
    n = len(tokens)
    display_table = [["" for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            if table[i][j]:
                display_table[i][j] = ", ".join(sorted(table[i][j]))
    headers = [f"{i+1}:{w}" for i, w in enumerate(tokens)]
    print("\nCYK Parse Table:\n")
    print(tabulate(display_table, headers=headers, tablefmt="grid"))

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    grammar_file = "/content/cyk_grammar.txt"  # Replace with your file path
    sentence = "astronomers saw stars with ears"
    tokens = sentence.split()

    grammar, rhs_to_lhs = read_grammar(grammar_file)
    table, back, tree_builder = cyk_parser(tokens, rhs_to_lhs)

    if tree_builder:
        tree_tuple = tree_builder(0, len(tokens)-1, 'S')
        print("\n✔ Sentence is valid.\n")
        nltk_tree = tuple_to_nltk_tree(tree_tuple)
        nltk_tree.pretty_print()
    else:
        print("\n✘ Sentence is invalid according to the grammar.\n")

    print_table(table,tokens)


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# Load dataset
df = pd.read_csv('/content/Machinetransalation.csv', encoding='latin1')

# Add <sos> and <eos> to target (French) text
df['french'] = df['french'].apply(lambda x: '<sos> ' + x + ' <eos>')

# Prepare source and target
input_texts = df['english'].tolist()
target_texts = df['french'].tolist()

# Tokenization
input_tokenizer = Tokenizer()
target_tokenizer = Tokenizer(filters='')
input_tokenizer.fit_on_texts(input_texts)
target_tokenizer.fit_on_texts(target_texts)

input_seq = input_tokenizer.texts_to_sequences(input_texts)
target_seq = target_tokenizer.texts_to_sequences(target_texts)

max_encoder_seq_length = max(len(seq) for seq in input_seq)
max_decoder_seq_length = max(len(seq) for seq in target_seq)

input_seq = pad_sequences(input_seq, maxlen=max_encoder_seq_length, padding='post')
target_seq = pad_sequences(target_seq, maxlen=max_decoder_seq_length, padding='post')

num_encoder_tokens = len(input_tokenizer.word_index) + 1
num_decoder_tokens = len(target_tokenizer.word_index) + 1

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(num_encoder_tokens, 256)(encoder_inputs)
encoder_lstm, state_h, state_c = LSTM(256, return_state=True)(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(num_decoder_tokens, 256)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Prepare target data
target_seq_input = target_seq[:, :-1]
target_seq_output = target_seq[:, 1:]
target_seq_output = np.expand_dims(target_seq_output, -1)

# Model checkpoint
checkpoint = ModelCheckpoint('seq2seq_model.h5', monitor='loss', save_best_only=True, mode='min', verbose=1)

# Training
model.fit([input_seq, target_seq_input], target_seq_output, batch_size=32, epochs=100, callbacks=[checkpoint])

# Encoder Model
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder Model
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
dec_emb2 = dec_emb_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)

# Reverse target tokenizer for decoding
reverse_target_word_index = dict((i, word) for word, i in target_tokenizer.word_index.items())

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index['<sos>']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index.get(sampled_token_index, '')

        if sampled_word == '<eos>' or len(decoded_sentence.split()) > max_decoder_seq_length:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]

    return decoded_sentence.strip()

# Test the model with a sentence
test_sentence = ['how are you']
test_seq = pad_sequences(input_tokenizer.texts_to_sequences(test_sentence), maxlen=max_encoder_seq_length, padding='post')
print(decode_sequence(test_seq))

#TEXTSUM
import numpy as np
import pandas as pd
import os
# Disable WandB logging
os.environ["WANDB_DISABLED"] = "true"

#TEXT FILE HANDLING - if it is a text file with article and summaries
# Read the text file with articles and summaries
# articles = []
# summaries = []

# with open("/content/articles_summaries.txt", "r", encoding="utf-8") as file:
#     for line in file:
#         # Assuming each line contains the article and summary separated by a tab or some other delimiter
#         article, summary = line.strip().split("\t")  # adjust split based on actual delimiter
#         articles.append(article)
#         summaries.append(summary)

# # Create a pandas dataframe from the lists
# import pandas as pd
# df = pd.DataFrame({"article": articles, "summary": summaries})

# # Convert to Hugging Face dataset
# from datasets import Dataset
# dataset = Dataset.from_pandas(df)


#TEXT FILE HANDLING - if it is a text file with ONLY ARTICLE
# # Read the text file with articles only
# articles = []

# with open("/content/articles_only.txt", "r", encoding="utf-8") as file:
#     for line in file:
#         articles.append(line.strip())  # Add each article to the list


# # Create a DataFrame from the articles list
# df = pd.DataFrame({"article": articles})

df = pd.read_csv("/content/news_summary.csv", encoding='utf-8', on_bad_lines='skip', engine='python')
df = df[["text", "headlines"]].dropna()
df = df.rename(columns={"text": "article", "headlines": "summary"})


from datasets import Dataset
from transformers import T5Tokenizer

dataset = Dataset.from_pandas(df)

tokenizer = T5Tokenizer.from_pretrained("t5-small")

def preprocess(example):
    input_text = "summarize: " + example["article"]
    target_text = example["summary"]

    input_enc = tokenizer(input_text, max_length=512, padding="max_length", truncation=True)
    target_enc = tokenizer(target_text, max_length=64, padding="max_length", truncation=True)

    input_enc["labels"] = target_enc["input_ids"]
    return input_enc

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

small_dataset = tokenized_dataset.select(range(100))  # Use 100 samples

from transformers import T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Load the model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5-finetuned-summary",
    per_device_train_batch_size=4,
    num_train_epochs=1,  # Reduce epochs to 1 for faster training
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="epoch",
    evaluation_strategy="no",
    report_to=None  # Ensure no WandB logging
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=small_dataset,
    tokenizer=tokenizer,
)
trainer.train()

#testing on a mannual article
input_text = "summarize: " + df.iloc[37]["article"]
input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids

output_ids = model.generate(input_ids, max_length=128, num_beams=8, early_stopping=True)
summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Original:", df.iloc[37]["article"])
print("Expected Summary:", df.iloc[37]["summary"])
print("Generated Summary:", summary)

#testing on a custom article
#new_article = "India launched a new space mission to study the Sun. This mission, called Aditya-L1, will provide data on solar radiation and storms."
# new_article = "Tesla reported record profits this quarter, driven by strong sales of the Model Y and Model 3. CEO Elon Musk highlighted the company's progress in AI and self-driving technology. Tesla also plans to open a new Gigafactory in India next year."
# new_article = "Scientists have discovered a new exoplanet that may be capable of supporting life. Located 120 light-years away, the planet is in the habitable zone of its star and has a similar atmosphere to Earth, with traces of oxygen and water vapor."
new_article = "The World Health Organization has issued a warning about a new strain of the flu virus spreading rapidly in Southeast Asia. Health officials urge people to get vaccinated and practice hygiene to avoid infection during the upcoming flu season."
# new_article = "In a thrilling final match, India won the ICC World Cup by defeating Australia. Virat Kohli scored a century, and Jasprit Bumrah took three crucial wickets. Fans celebrated the victory with parades and fireworks across the country."
# new_article = "Apple has announced the launch of its latest iPhone 15 series. The new iPhones come with an upgraded A17 Bionic chip, improved cameras, and USB-C charging. The Pro models also feature a titanium frame, making them lighter and more durable."

input_text = "summarize: " + new_article
input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids

output_ids = model.generate(input_ids, max_length=128, num_beams=8, early_stopping=True)
summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Custom Summary:", summary)

#no fine tuning just summarize
df_new = pd.read_csv("/content/news_summary.csv")
df_new = df_new[["text"]].dropna()
df_new = df_new.rename(columns={"text": "article"})


from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

for i in range(5):  # Just first 5 articles for demo
    input_text = "summarize: " + df_new.iloc[i]["article"]
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids
    output_ids = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"\n📰 Article {i+1}:\n", df_new.iloc[i]["article"])
    print("📝 Generated Summary:\n", summary)


#SENTIMENT
from transformers import pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

text = "The movie was absolutely amazing, I loved every part of it!"
result = sentiment_analyzer(text)

print("Text:", text)
print("Sentiment:", result)

text = "The movie was absolutely boring,Disappointed"
result = sentiment_analyzer(text)

print("Text:", text)
print("Sentiment:", result)

#using vaccine dataset, can use artcle also (check for artcle column alone)
from transformers import pipeline
import pandas as pd
import re

# Load the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Load the dataset
sentiment_df = pd.read_csv("/content/vaccination_tweets.csv")  # Replace with your file path
tweets = sentiment_df['text'].values

# Preprocessing function for cleaning tweets
def data_preprocess(words):
    # Removing any emojis or unknown characters
    words = words.encode('ascii', 'ignore')
    words = words.decode()

    # Splitting string into words
    words = words.split(' ')

    # Removing URLs
    words = [word for word in words if not word.startswith('http')]
    words = ' '.join(words)

    # Removing punctuations
    words = re.sub(r"[^0-9a-zA-Z]+", " ", words)

    # Removing extra spaces
    words = re.sub(' +', ' ', words)
    return words

# Preprocess the tweets
preprocessed_tweets = [data_preprocess(tweet) for tweet in tweets]

# Create a DataFrame with preprocessed tweets
sentiment_df['preprocessed_text'] = preprocessed_tweets

# Take the first 100 rows from the sentiment_df
sample_df = sentiment_df.head(100).copy()

# Apply sentiment analysis to the preprocessed tweets and store results in a new column
sample_df["sentiment"] = [sentiment_analyzer(tweet)[0]['label'] for tweet in sample_df['preprocessed_text']]

# Display the first few results
print(sample_df[["preprocessed_text", "sentiment"]].head(10))


#model= nu potrunga sentiment_analyzer = pipeline( "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
# TRAINING BLOCK
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

# Load IMDb dataset
dataset = load_dataset("imdb")

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Tokenize text
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="no",
    num_train_epochs=0.5,  # quick test
    per_device_train_batch_size=4,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    load_best_model_at_end=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].select(range(200)),
    eval_dataset=tokenized_datasets["test"].select(range(100)),
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./results")  # Save for later use

# Evaluate
results = trainer.evaluate()
print("Evaluation results:", results)

# INFERENCE BLOCK
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the fine-tuned model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("./results")  # load trained model

# Inference: Predict sentiment of a new review
def predict_sentiment(text):
    model.eval()  # set model to evaluation mode

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)

    # Get model's output (logits)
    with torch.no_grad():  # avoid tracking gradients
        outputs = model(**inputs)

    # Get predicted probabilities (softmax)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Get predicted label (0 = negative, 1 = positive)
    prediction = torch.argmax(probs, dim=-1).item()

    # Return prediction
    return "positive" if prediction == 1 else "negative"

# Test with some new examples
test_texts = [
    "The movie was fantastic! I really loved it.",
    "It was the worst movie I have ever seen, so boring.",
    "A wonderful experience with great acting and storyline.",
    "Terrible! I would not recommend it to anyone."
]

for text in test_texts:
    sentiment = predict_sentiment(text)
    print(f"Review: {text}")
    print(f"Predicted Sentiment: {sentiment}\n")
    
    
#machine-transformer
from transformers import MarianMTModel, MarianTokenizer

def translate(text, src_lang, tgt_lang):
    # Create the model name based on source and target languages
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

    try:
        # Load tokenizer and model
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
    except Exception as e:
        return f" Error loading model for {src_lang} to {tgt_lang}: {e}"

    # Tokenize input text
    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Generate translation
    translated = model.generate(**encoded)

    # Decode the output
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# 🔄 Interactive use
print("🌐 Dynamic Machine Translation")
src = input("Enter source language code (e.g., en, fr, de, es): ").strip()
tgt = input("Enter target language code (e.g., fr, en, de, es): ").strip()
text = input("Enter text to translate:\n")

translation = translate(text, src, tgt)
print(f"\n Original ({src}): {text}")
print(f" Translated ({tgt}): {translation}")

#finetuning
#!pip install transformers datasets sentencepiece -q
from datasets import load_dataset
from transformers import MarianTokenizer, MarianMTModel, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Load a small en-fr dataset
dataset = load_dataset("opus_books", "en-fr")
small_data = dataset["train"].select(range(100))  # Use just 100 examples

# Load tokenizer and model
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Preprocess function
def preprocess(example):
    src = example["translation"]["en"]
    tgt = example["translation"]["fr"]
    model_inputs = tokenizer(src, max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(tgt, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize dataset
tokenized = small_data.map(preprocess, batched=False)

# Training arguments (FAST!)
args = Seq2SeqTrainingArguments(
    output_dir="./quick_mt_model",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    save_strategy="no",
    logging_steps=10,
    report_to=None,
    fp16=False  # If on GPU with FP16 support, set True
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer
)

# Train (~2–4 mins)
trainer.train()

def translate_custom(text):
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, padding=True).input_ids
    output_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Example test
print(translate_custom("The forest was dark and silent."))


# READING CSV OR TEXT FILE

# # Read the text file
# src_sentences = []
# tgt_sentences = []

# with open('text_file.txt', 'r', encoding='utf-8') as file:
#     for line in file:
#         src, tgt = line.strip().split('\t')  # Assuming tab-separated
#         src_sentences.append(src)
#         tgt_sentences.append(tgt)

# # Create a DataFrame and convert to Hugging Face Dataset
# df = pd.DataFrame({"src_text": src_sentences, "tgt_text": tgt_sentences})
# dataset = Dataset.from_pandas(df)


# #READING CSV FILE
# # Load your CSV into a DataFrame
# df = pd.read_csv('csv_file.csv')  # Update with your file path

# # Rename columns for easier access
# df = df.rename(columns={"en": "src_text", "fr": "tgt_text"})

# # Convert DataFrame to Hugging Face Dataset format
# dataset = Dataset.from_pandas(df)


#PCYK
from collections import defaultdict
import math
import pprint
from prettytable import PrettyTable

# Step 1: Hardcoded parse trees from the image
tree1 = ['S',
           ['NP', 'John'],
           ['VP',
               ['VP', ['V', 'called'], ['NP', 'Mary']],
               ['PP', ['P', 'from'], ['NP', 'Denver']]
           ]
        ]

tree2 = ['S',
           ['NP', 'John'],
           ['VP',
               ['V', 'called'],
               ['NP',
                   ['NP', 'Mary'],
                   ['PP', ['P', 'from'], ['NP', 'Denver']]
               ]
           ]
        ]

# Step 2: Extract productions
def extract_productions(tree, productions):
    if isinstance(tree, list):
        lhs = tree[0]
        rhs = []

        for child in tree[1:]:
            if isinstance(child, list):
                rhs.append(child[0])
                extract_productions(child, productions)
            else:
                rhs.append(child)
        rule = (lhs, tuple(rhs))
        productions[rule] += 1

productions = defaultdict(int)
extract_productions(tree1, productions)
extract_productions(tree2, productions)

# Step 3: Compute PCFG
lhs_counts = defaultdict(int)
for (lhs, rhs), count in productions.items():
    lhs_counts[lhs] += count

pcfg = defaultdict(list)
for (lhs, rhs), count in productions.items():
    prob = count / lhs_counts[lhs]
    pcfg[lhs].append((rhs, prob))

print("\n--- PCFG ---")
for lhs, rules in pcfg.items():
    for rhs, prob in rules:
        print(f"{lhs} -> {' '.join(rhs)} [{prob:.2f}]")


# Step 4: Viterbi parser for new sentence
sentence = ['John', 'called', 'Mary', 'from', 'Denver']
n = len(sentence)
table = [[defaultdict(lambda: (-math.inf, None)) for _ in range(n)] for _ in range(n)]

# Step 5: Initialize terminals
for i, word in enumerate(sentence):
    for lhs, rules in pcfg.items():
        for rhs, prob in rules:
            if len(rhs) == 1 and rhs[0] == word:
                table[i][i][lhs] = (math.log(prob), word)

# Step 6: CKY-style Viterbi algorithm
for span in range(2, n+1):
    for i in range(n - span + 1):
        j = i + span - 1
        for k in range(i, j):
            for lhs, rules in pcfg.items():
                for rhs, prob in rules:
                    if len(rhs) == 2:
                        B, C = rhs
                        if B in table[i][k] and C in table[k+1][j]:
                            prob_B, back_B = table[i][k][B]
                            prob_C, back_C = table[k+1][j][C]
                            total_prob = math.log(prob) + prob_B + prob_C
                            if total_prob > table[i][j][lhs][0]:
                                table[i][j][lhs] = (total_prob, (k, B, C))

# Step 7: Backtrack to recover tree
def build_tree(i, j, symbol):
    prob, back = table[i][j].get(symbol, (-math.inf, None))

    # Terminal rule: back is a string (the word)
    if isinstance(back, str):
        return [symbol, back]

    # Unary case fallback (not expected here, but safe)
    if back is None:
        return [symbol]

    # Binary rule: back = (k, B, C)
    k, B, C = back
    left = build_tree(i, k, B)
    right = build_tree(k+1, j, C)
    return [symbol, left, right]

# Final output
print("\n--- Most Probable Parse Tree ---")
if 'S' in table[0][n-1]:
    tree = build_tree(0, n-1, 'S')
    pprint.pprint(tree)

    print(f"\nProbability of the parse tree: {math.exp(table[0][n-1]['S'][0]):.8f}")

    print("\n--- Viterbi Parsing Table (Triangular Format) ---\n")

    pretty_table = []

    for row in range(n):
        current_row = []
        for col in range(n):
            if col < row:
                current_row.append("")  # lower triangle blank
            else:
                cell = table[row][col]
                if not cell:
                    current_row.append("-")
                else:
                    entries = []
                    for symbol, (prob, back) in sorted(cell.items(), key=lambda x: -x[1][0]):
                        prob_val = math.exp(prob)
                        if isinstance(back, str):
                            entries.append(f"{symbol}({prob_val:.2f})")
                        else:
                            entries.append(f"{symbol}({prob_val:.6f})")
                    current_row.append("\n".join(entries))
        pretty_table.append(current_row)

    # Create header row
    headers = [""] + sentence
    cyk = PrettyTable()
    cyk.field_names = headers
    for i, row in enumerate(pretty_table):
        cyk.add_row([sentence[i]] + row)

    print(cyk)

else:
    print("No valid parse found.")

# IF YOU HAVE 5-6 TREES IN QUESTION PAPER, DO LIKE THIS
#LOOP
# trees = [
#     ['S',
#         ['NP', 'John'],
#         ['VP',
#             ['VP', ['V', 'called'], ['NP', 'Mary']],
#             ['PP', ['P', 'from'], ['NP', 'Denver']]
#         ]
#     ],
#     ['S',
#         ['NP', 'John'],
#         ['VP',
#             ['V', 'called'],
#             ['NP',
#                 ['NP', 'Mary'],
#                 ['PP', ['P', 'from'], ['NP', 'Denver']]
#             ]
#         ]
#     ],
#     # Add 4 more trees here:
#     tree3,
#     tree4,
#     tree5,
#     tree6,
# ]

# CHANGE IN THIS PART

# productions = defaultdict(int)

# for tree in trees:
#     extract_productions(tree, productions)


# #TO CHECK PRODUCTIONS
# print("\n--- Production Counts ---")
# for (lhs, rhs), count in productions.items():
#     print(f"{lhs} -> {' '.join(rhs)} : {count}")

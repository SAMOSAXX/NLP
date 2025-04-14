#example
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
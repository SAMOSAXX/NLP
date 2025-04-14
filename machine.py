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

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

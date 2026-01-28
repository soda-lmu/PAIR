import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score  # Import f1_score
from scipy.special import softmax

# 1. Prepare the data
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)  # Change to long type
        }

mode = "adj"# TODO: select from ['bal', 'unbal1', 'unbal2', 'adj']
X = 0.05 # TODO: select from [0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3]
train_l = "new_hs" # TODO: select from ['new_hs', 'new_ol']

# Read Data
df = pd.read_csv(f"data/all_labels_full.csv", index_col=0)
# select the dataset mode
df_dataset = df[df['dataset'].str.contains(mode)]
# further select based on column x which contains X value
df_dataset = df_dataset[df_dataset[X].notna()]

train = df_dataset[(df_dataset['tweet.id'] >= 1) & (df_dataset['tweet.id'] <= 2000)]
val = df_dataset[(df_dataset['tweet.id'] >= 2001) & (df_dataset['tweet.id'] <= 2500)]
test = df[df['dataset'].str.contains("gold")]

train_texts = list(train.tweet_hashed)
val_texts = list(val.tweet_hashed)
test_texts = list(test.tweet_hashed)

train_labels = list(train[train_l])
val_labels = list(val[train_l])
test_labels = list(test["hs" if train_l == "new_hs" else "ol"])

# Initialize the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
max_length = 128

# Create training and validation datasets
train_dataset = TweetDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = TweetDataset(val_texts, val_labels, tokenizer, max_length)
test_dataset = TweetDataset(test_texts, test_labels, tokenizer, max_length)

# 2. Define the RoBERTa classification model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# 3. Set training parameters
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=500,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    load_best_model_at_end=True,
    metric_for_best_model='f1',  # Choose F1 score as the metric
    greater_is_better=True,      # Because a higher F1 score is better
)

# Custom evaluation metric
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)  # Get the index of the max value as prediction
    accuracy = (preds == labels).mean()
    f1 = f1_score(labels, preds)
    return {'accuracy': accuracy, 'f1': f1}

# 4. Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset, 
    compute_metrics=compute_metrics
)

trainer.train()

# 5. Evaluate the model and get logits

test_results = trainer.predict(test_dataset)
logits = test_results.predictions
labels = test_results.label_ids

# use softmax function to convert logits to probabilities
probabilities = softmax(logits, axis=-1)

# select the probability for class 1 as the final probability
probabilities_class_1 = probabilities[:, 1]

# transform probabilities to list
probability_list = probabilities_class_1.tolist()

# get the predictions
predictions = logits.argmax(-1)

test[f"{mode}_preds"] = predictions
test[f"{mode}_preds_scores"] = probability_list

# save the test results to a tempfile. After all results are collected, we will merge them.
test.to_csv(f"test_results/test_{mode}_label_{X}_{train_l}.csv")

print(test_results.metrics)


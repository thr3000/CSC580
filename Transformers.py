#Importing Modules
import collections

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm
import transformers
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

import csv
import pandas as pd

from sklearn.model_selection import train_test_split
import torchtext.data.utils as utils
from datasets import Dataset


##intial random seeds#
seed = 1234

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


#demon datasets#
#train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])
#print("dataset",train_data, test_data)
#print("feature",train_data.features)
#print("data 0 ",train_data[0])

#Loading Bitcoin data
# Load data from CSV file
data = pd.read_csv('combined_without_other.csv')

#Drop rows where 'text' is NaN
data = data.dropna(subset=['text'])
data = data.rename(columns={'sentiment': 'label'})
data['label'] = data['label'].replace({'Positive': 1, 'Negative': 0})

label_counts = data['label'].value_counts()
print(label_counts)
#print(data['label'].value_counts())

#drop data is not 0 and 1 
data = data.dropna(subset=['label'])
data = data[data['label'].isin([0, 1])]


#experiment
# Extract only the first 10 rows
#data = data.head(250000)
#data_test = data.head(250000).copy(deep=True)
#print(data_test['label'].value_counts())


#pdb.set_trace()


# Display the DataFrame
print(data)

# Split the data into training and test sets
train_data_df, test_data_df = train_test_split(data, test_size=0.2, random_state=42)


# Convert DataFrame to datasets.Dataset
train_data = Dataset.from_pandas(train_data_df)
test_data = Dataset.from_pandas(test_data_df)






#Tokenization#
transformer_name = "bert-base-uncased"
tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_name)

#print(tokenizer.tokenize("hello world!"))
#print(tokenizer.encode("hello world!"))
#print(tokenizer.convert_ids_to_tokens(tokenizer.encode("hello world")))
#print(tokenizer("hello world!"))

def tokenize_and_numericalize_example(example, tokenizer):
    ids = tokenizer(example["text"], truncation=True)["input_ids"]
    return {"ids": ids}

train_data = train_data.map(
    tokenize_and_numericalize_example, fn_kwargs={"tokenizer": tokenizer}
)
test_data = test_data.map(
    tokenize_and_numericalize_example, fn_kwargs={"tokenizer": tokenizer}
)


#print(train_data[0])
#print(tokenizer.vocab["!"])
#print(tokenizer.pad_token)
#print(tokenizer.pad_token_id)
#print(tokenizer.vocab[tokenizer.pad_token])

pad_index = tokenizer.pad_token_id


#create validation data#
test_size = 0.25

train_valid_data = train_data.train_test_split(test_size=test_size)
train_data = train_valid_data["train"]
valid_data = train_valid_data["test"]

train_data = train_data.with_format(type="torch", columns=["ids", "label"])
valid_data = valid_data.with_format(type="torch", columns=["ids", "label"])
test_data = test_data.with_format(type="torch", columns=["ids", "label"])




#create dataload
def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "label": batch_label}
        return batch

    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader

batch_size = 64

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)


#model
class Transformer(nn.Module):
    def __init__(self, transformer, output_dim, freeze):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.config.hidden_size
        self.fc = nn.Linear(hidden_dim, output_dim)
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, ids):
        # ids = [batch size, seq len]
        output = self.transformer(ids, output_attentions=True)
        hidden = output.last_hidden_state
        # hidden = [batch size, seq len, hidden dim]
        attention = output.attentions[-1]
        # attention = [batch size, n heads, seq len, seq len]
        cls_hidden = hidden[:, 0, :]
        prediction = self.fc(torch.tanh(cls_hidden))
        # prediction = [batch size, output dim]
        return prediction


transformer = transformers.AutoModel.from_pretrained(transformer_name)
#print(transformer.config.hidden_size)


output_dim = len(train_data["label"].unique())
freeze = False

model = Transformer(transformer, output_dim, freeze)



#Pre-trainning process setting 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"The model has {count_parameters(model):,} trainable parameters")
#optimizer
lr = 1e-5
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


#sending data to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)
criterion = criterion.to(device)
#pdb.set_trace()



#trainning
"""
def train(data_loader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(data_loader, desc="training..."):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)
        prediction = model(ids)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)
def evaluate(data_loader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)
"""
def train(data_loader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    total_batches = len(data_loader)
    for batch_index, batch in enumerate(data_loader):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)
        prediction = model(ids)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())

        if (batch_index + 1) % 10 == 0:  # Print every 10 batches
            print(f"Training: Batch {batch_index+1}/{total_batches}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")
    
    return np.mean(epoch_losses), np.mean(epoch_accs)

def evaluate(data_loader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    total_batches = len(data_loader)
    for batch_index, batch in enumerate(data_loader):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)
        prediction = model(ids)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())

        if (batch_index + 1) % 10 == 0:  # Print every 10 batches
            print(f"Evaluating: Batch {batch_index+1}/{total_batches}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")
    
    return np.mean(epoch_losses), np.mean(epoch_accs)
def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy



n_epochs = 5
best_valid_loss = float("inf")

metrics = collections.defaultdict(list)

for epoch in range(n_epochs):
    train_loss, train_acc = train(
        train_data_loader, model, criterion, optimizer, device
    )
    valid_loss, valid_acc = evaluate(valid_data_loader, model, criterion, device)
    metrics["train_losses"].append(train_loss)
    metrics["train_accs"].append(train_acc)
    metrics["valid_losses"].append(valid_loss)
    metrics["valid_accs"].append(valid_acc)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "transformer.pt")
    print(f"epoch: {epoch}")
    print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
    print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(metrics["train_losses"], label="train loss")
ax.plot(metrics["valid_losses"], label="valid loss")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_xticks(range(n_epochs))
ax.legend()
ax.grid()
fig.savefig("transformers_loss.png")


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(metrics["train_accs"], label="train accuracy")
ax.plot(metrics["valid_accs"], label="valid accuracy")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_xticks(range(n_epochs))
ax.legend()
ax.grid()
fig.savefig("transformers_accuracy.png")



model.load_state_dict(torch.load("transformer.pt"))
test_loss, test_acc = evaluate(test_data_loader, model, criterion, device)

print(f"test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}")



#test for new content
def predict_sentiment(text, model, tokenizer, device):
    ids = tokenizer(text)["input_ids"]
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    return predicted_class, predicted_probability

print(model)

text = "This film is terrible!"
print(predict_sentiment(text, model, tokenizer, device))


#print(predict_sentiment("This film is terrible!", model, tokenizer, device))
pdb.set_trace()
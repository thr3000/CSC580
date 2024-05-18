import collections
import torch
import torch.nn as nn
import torch.nn.functional as F

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm
import pdb

from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

import csv
import pandas as pd

from sklearn.model_selection import train_test_split
import torchtext.data.utils as utils
from datasets import Dataset




#import gensim

# Load Word2Vec embeddings
# model = gensim.models.KeyedVectors.load_word2vec_format('path_to_word2vec.bin', binary=True)

def load_glove_model(glove_file_path):
    print("Loading GloVe Model")
    glove_model = {}
    with open(glove_file_path, 'r', encoding="utf8") as f:
        for line in f:
            split_lines = line.split()
            word = split_lines[0]
            embedding = [float(value) for value in split_lines[1:]]
            glove_model[word] = embedding
    print("Loaded GloVe Model with %s words." % len(glove_model))
    return glove_model

glove_path = "glove.6B.100d.txt"  # Adjust the path if your file is located in a different folder
glove_model = load_glove_model(glove_path)

vector = glove_model.get("computer")  # Replace "computer" with any word
if vector:
    print("Vector for 'computer':", vector)
else:
    print("Word not in vocabulary.")





##intial random seeds#
seed = 1234

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True



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

#pdb.set_trace()

#print(train_data.head())  # Print the first few rows of the training set
#print(test_data.head())  # Print the first few rows of the test set


# Define the tokenizer
tokenizer = torchtext.data.utils.get_tokenizer("basic_english")



# Define a function to tokenize and truncate
#def tokenize_and_truncate(text, tokenizer, max_length):
#    text = str(text)
#    tokens = tokenizer(text)
#    truncated_tokens = tokens[:max_length]
#    return truncated_tokens
def tokenize_example(example, tokenizer, max_length):
    tokens = tokenizer(example["text"])[:max_length]
    length = len(tokens)
    return {"tokens": tokens, "length": length}



#if one token more than max_length Truncation/Splitting/Error 
max_length = 256
#to tokenization the test data with each example in dataset
print("start train_data maping")
train_data = train_data.map(
    tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
)
print("start test_data maping")
test_data = test_data.map(
    tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
)


#pdb.set_trace()
#create validation data
test_size = 0.25

train_valid_data = train_data.train_test_split(test_size=test_size)
train_data = train_valid_data["train"]
valid_data = train_valid_data["test"]
#print(len(train_data), len(valid_data), len(test_data))



min_freq = 5
special_tokens = ["<unk>", "<pad>"]

vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)



###operation tool box###
#print(len(vocab))
#print(vocab.get_itos()[:10])
#print(vocab["or"])
unk_index = vocab["<unk>"]
pad_index = vocab["<pad>"]
#print(unk_index, pad_index)
#print("or" in vocab) #true / False


#to pass the unknow new words, when face unknow word "false--> 0"
vocab.set_default_index(unk_index)
#print(vocab["some_token"])
#print(vocab.lookup_indices(["hello", "world", "some_token", "<pad>"]))


##Numericalizing Data##
"""
we'll define a function that takes an example and our vocabulary, gets the index for
each token in each example and then creates an ids field which containes the numericalized tokens.
"""
#for each tokens(words) label a specifci "id" - set "ids"
def numericalize_example(example, vocab):
    ids = vocab.lookup_indices(example["tokens"])
    return {"ids": ids}
train_data = train_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
valid_data = valid_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
test_data = test_data.map(numericalize_example, fn_kwargs={"vocab": vocab})


##after treatment  we can find each data by number ""tool box"
#print(train_data[0]["tokens"][:10])
#print("vocab.lookup_indices",vocab.lookup_indices(train_data[0]["tokens"][:10])) #a
#print("train_data[0][ids]  ",train_data[0]["ids"][:10]) #b   a should = b


#transforming the ids and label from integers into PyTorch tensors
#train_data = train_data.with_format(type="torch", columns=["ids", "label"])
#valid_data = valid_data.with_format(type="torch", columns=["ids", "label"])
#test_data = test_data.with_format(type="torch", columns=["ids", "label"])
train_data = train_data.with_format(type="torch", columns=["ids", "label", "length"])
valid_data = valid_data.with_format(type="torch", columns=["ids", "label", "length"])
test_data = test_data.with_format(type="torch", columns=["ids", "label", "length"])



#pdb.set_trace()



##create Data Loaders##
def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        batch_length = [i["length"] for i in batch]
        batch_length = torch.stack(batch_length)
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "length": batch_length, "label": batch_label}
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

batch_size = 512

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)


# Function to print data from a DataLoader
def print_data_loader_contents(data_loader, num_batches=1):
    for i, batch in enumerate(data_loader):
        print(f"Batch {i + 1}:")
        ids = batch['ids']        # Assuming your collate function packs them as 'ids'
        labels = batch['label']   # Assuming labels are stored under 'label'
        lengths = batch['length'] # Assuming lengths are stored under 'length'

        # Print the actual data in this batch
        print("IDs:", ids)
        print("Labels:", labels)
        print("Lengths:", lengths)
        print("\n")

        # Stop after printing `num_batches` batches
        if i >= num_batches - 1:
            break

# Example usage:
print("Training Data Loader Contents:")
print_data_loader_contents(train_data_loader, num_batches=2)





"""
#model LSTM
class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout_rate,
        pad_index,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        # ids = [batch size, seq len]
        # length = [batch size]
        embedded = self.dropout(self.embedding(ids))
        # embedded = [batch size, seq len, embedding dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, length, batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output = [batch size, seq len, hidden dim * n directions]
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
            # hidden = [batch size, hidden dim * 2]
        else:
            hidden = self.dropout(hidden[-1])
            # hidden = [batch size, hidden dim]
        prediction = self.fc(hidden)
        # prediction = [batch size, output dim]
        return prediction

vocab_size = len(vocab)
embedding_dim = 300
hidden_dim = 300
output_dim = len(train_data.unique("label"))
n_layers = 2
bidirectional = True
dropout_rate = 0.5

model = LSTM(
    vocab_size,
    embedding_dim,
    hidden_dim,
    output_dim,
    n_layers,
    bidirectional,
    dropout_rate,
    pad_index,
)
"""

class EnhancedLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout_rate,
        pad_index,
        pretrained_embeddings=None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)

    def forward(self, ids, lengths):
        # Embedding and dropout
        embedded = self.dropout(self.embedding(ids))

        # Pack the sequences for RNN processing
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # Unpack the sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Attention mechanism: we'll calculate weights and then apply these to the LSTM output
        attention_weights = torch.softmax(self.attention(output), dim=1)  # [batch_size, seq_len, 1]
        attention_output = torch.bmm(attention_weights.transpose(1, 2), output).squeeze(1)  # [batch_size, hidden_dim * num_directions]

        # Pass through the fully connected layer
        prediction = self.fc(self.dropout(attention_output))
        return prediction

vocab_size = len(vocab)
embedding_dim = 300
hidden_dim = 300
output_dim = len(train_data.unique("label"))
n_layers = 3
bidirectional = True
dropout_rate = 0.5


# Model instantiation part remains the same, but you can load pre-trained embeddings if available
model = EnhancedLSTM(
    vocab_size,
    embedding_dim,
    hidden_dim,
    output_dim,
    n_layers,
    bidirectional,
    dropout_rate,
    pad_index,
    pretrained_embeddings=None,  # Replace None with your pre-trained embedding tensor if available
)




#count totally parameter
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"The model has {count_parameters(model):,} trainable parameters")


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.orthogonal_(param)
print(model.apply(initialize_weights))


#from torchtext.vocab import GloVe
#glove_vectors = GloVe(name='6B', dim=100)

#vectors = torchtext.vocab.GloVe()
#pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
lr = 5e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(device)
criterion = criterion.to(device)


#define trainning
"""
def train(dataloader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(dataloader, desc="training..."):
        ids = batch["ids"].to(device)
        length = batch["length"]
        label = batch["label"].to(device)
        prediction = model(ids, length)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)
def evaluate(dataloader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            length = batch["length"]
            label = batch["label"].to(device)
            prediction = model(ids, length)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)
"""
def train(dataloader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    total_batches = len(dataloader)
    for batch_index, batch in enumerate(dataloader):
        ids = batch["ids"].to(device)
        length = batch["length"]
        label = batch["label"].to(device)
        prediction = model(ids, length)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
        
        if (batch_index + 1) % 10 == 0:  # Print every 10 batches
            print(f"Training: Batch {batch_index+1}/{total_batches}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")
    
    avg_loss = np.mean(epoch_losses)
    avg_acc = np.mean(epoch_accs)
    print(f"Training: Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_acc:.4f}")
    return avg_loss, avg_acc

def evaluate(dataloader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    total_batches = len(dataloader)
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            ids = batch["ids"].to(device)
            length = batch["length"]
            label = batch["label"].to(device)
            prediction = model(ids, length)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())

            if (batch_index + 1) % 10 == 0:  # Print every 10 batches
                print(f"Evaluating: Batch {batch_index+1}/{total_batches}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

    avg_loss = np.mean(epoch_losses)
    avg_acc = np.mean(epoch_accs)
    print(f"Evaluating: Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_acc:.4f}")
    return avg_loss, avg_acc

def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


#trainning process
n_epochs = 30
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
        torch.save(model.state_dict(), "lstm.pt")
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
fig.savefig("loss_.png")

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(metrics["train_accs"], label="train accuracy")
ax.plot(metrics["valid_accs"], label="valid accuracy")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_xticks(range(n_epochs))
ax.legend()
ax.grid()
fig.savefig("accuracy_.png")


#testing prcess
model.load_state_dict(torch.load("lstm.pt"))
test_loss, test_acc = evaluate(test_data_loader, model, criterion, device)
print(f"test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}")


torch.save(model.state_dict(), "lstm.pt")




#pridiction tool for new content
def predict_sentiment(text, model, tokenizer, vocab, device):
    tokens = tokenizer(text)
    ids = vocab.lookup_indices(tokens)
    length = torch.LongTensor([len(ids)])
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor, length).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    output_class = "negative" if predicted_class == 0 else "positive"
    return output_class,predicted_class, predicted_probability

text = "This film is terrible!"
print(predict_sentiment(text, model, tokenizer, vocab, device))
#print(predict_sentiment("netflix and chill tonight?", model, tokenizer, vocab, device))


# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples
from sentiment_data import read_word_embeddings, read_word_embeddings_random, read_word_embeddings_random_BPE
from torch.utils.data import Dataset
import numpy as np


# Dataset class for handling sentiment analysis data
class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, emb_file, vectorizer=None, train=True):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)
        
        # Extract sentences and labels from the examples
        self.sentences = [" ".join(ex.words) for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]
        
        word_embeddings = read_word_embeddings(emb_file)
        
        # Convert embeddings and labels to PyTorch tensors
        ex_embedding = []
        for ex in self.examples:
            emb_temp = []
            for word in ex.words:
                emb_temp.append(word_embeddings.get_embedding(word))
            embeddings_array = np.array(emb_temp)
            average_embedding = np.mean(embeddings_array, axis=0)
            ex_embedding.append(average_embedding)
            
        self.embeddings = torch.tensor(ex_embedding, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the feature vector and label for the given index
        return self.embeddings[idx], self.labels[idx]
    

# Two-layer fully connected neural network
class NN2DAN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x
    
class NN3DAN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.log_softmax(x)

class SentimentDatasetRandomDAN(Dataset):
    def __init__(self, infile, emb_file, vectorizer=None, train=True):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)
        
        # Extract sentences and labels from the examples
        self.sentences = [" ".join(ex.words) for ex in self.examples]
        self.labels = [torch.tensor(ex.label, dtype = torch.long) for ex in self.examples]

        self.word_embeddings = read_word_embeddings_random(emb_file)
        self.input_size = len(self.word_embeddings.word_indexer)

        # Convert embeddings and labels to PyTorch tensors
        total_ind = []
        for ex in self.examples:
            ind_temp = []
            for word in ex.words:
                word_idx = self.word_embeddings.word_indexer.index_of(word)
                if word_idx == -1:
                    word_idx = self.word_embeddings.word_indexer.index_of("UNK")
                    
                ind_temp.append(word_idx)
            total_ind.append(torch.tensor(ind_temp, dtype = torch.long))
        
        # total_ind = np.array(total_ind)
        self.ind = total_ind
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the feature vector and label for the given index
        return self.ind[idx], self.labels[idx]
    
    def collate_fn(self, batch):
        embeddings, labels = zip(*batch)
        max_len = max([len(embedding) for embedding in embeddings])
        padded_embeddings = []
        for embedding in embeddings:
            padding = torch.zeros((max_len - len(embedding)))
            padded_embedding = torch.cat((embedding, padding), 0)
            padded_embeddings.append(padded_embedding)
        return torch.stack(padded_embeddings), torch.stack(labels)


class NN2RandomDAN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NN2RandomDAN, self).__init__()
        self.emb_layer = torch.nn.Embedding(input_size,hidden_size, padding_idx = 0)  # Use the embedding layer from the dataset
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # Set input size based on embedding size
        self.fc2 = nn.Linear(hidden_size, 2)  # Assuming binary classification
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.long()
        x = self.emb_layer(x)  # Get embeddings for the indices
        x = torch.mean(x, dim = 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x
    
class NN3RandomDAN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NN3RandomDAN, self).__init__()
        self.emb_layer = torch.nn.Embedding(input_size,hidden_size, padding_idx = 0)  # Use the embedding layer from the dataset
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # Set input size based on embedding size
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Set input size based on embedding size
        self.fc3 = nn.Linear(hidden_size, 2)  # Assuming binary classification
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.long()
        x = self.emb_layer(x)  # Get embeddings for the indices
        x = torch.mean(x, dim = 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.log_softmax(x)
        return x


class SentimentDatasetRandomBPE(Dataset):
    def __init__(self, infile, emb_file, vocab, vectorizer=None, train=True):
        self.vocab = vocab

        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)

        # Extract sentences and labels from the examples
        self.sentences = [" ".join(ex.words) for ex in self.examples]
        self.labels = [torch.tensor(ex.label, dtype = torch.long) for ex in self.examples]

        self.subword_embeddings = read_word_embeddings_random_BPE(self.vocab)
        self.input_size = len(self.subword_embeddings.word_indexer)
        
        # Convert embeddings to PyTorch tensors
        self.ind = self.break_sentences_into_subwords(self.vocab)
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the feature vector and label for the given index
        return self.ind[idx], self.labels[idx]
    
    def break_sentences_into_subwords(self, vocab):
        """Break a batch of sentences into subwords based on the given vocabulary."""
        all_subword_indices = []
        
        for sentence in self.sentences:
            words = sentence.split()
            subwords = []
            
            for word in words:
                while word:
                    # Find the longest subword in the vocabulary
                    for i in range(len(word), 0, -1):
                        subword = word[:i]
                        if subword in vocab:
                            subwords.append(self.subword_embeddings.word_indexer.index_of(subword))
                            word = word[i:]  # Remove the matched subword from the word
                            break
                    else:
                        break
            all_subword_indices.append(torch.tensor(subwords, dtype = torch.long))

        return all_subword_indices

    def collate_fn(self, batch):
        embeddings, labels = zip(*batch)
        max_len = max([len(embedding) for embedding in embeddings])
        padded_embeddings = []
        for embedding in embeddings:
            padding = torch.zeros((max_len - len(embedding)))
            padded_embedding = torch.cat((embedding, padding), 0)
            padded_embeddings.append(padded_embedding)
        return torch.stack(padded_embeddings), torch.stack(labels)


# Two-layer fully connected neural network
class NN2RandomBPE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NN2RandomBPE, self).__init__()
        self.emb_layer = torch.nn.Embedding(input_size,hidden_size, padding_idx = 0)  # Use the embedding layer from the dataset
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # Set input size based on embedding size
        self.fc2 = nn.Linear(hidden_size, 2)  # Assuming binary classification
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.long()
        x = self.emb_layer(x)  # Get embeddings for the indices
        x = torch.mean(x, dim = 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x

class NN3RandomBPE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NN3RandomBPE, self).__init__()
        self.emb_layer = torch.nn.Embedding(input_size,hidden_size, padding_idx = 0)  # Use the embedding layer from the dataset
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # Set input size based on embedding size
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)  # Assuming binary classification
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.long()
        x = self.emb_layer(x)  # Get embeddings for the indices
        x = torch.mean(x, dim = 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.log_softmax(x)
        return x







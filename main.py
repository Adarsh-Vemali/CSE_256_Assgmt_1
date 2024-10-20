# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import *
from utils import *
from collections import Counter, defaultdict

# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)

        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')
    parser.add_argument('--glove_size', type=str, required=False, default='300', help='Model type to train (e.g., BOW)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load dataset
    start_time = time.time()

    train_data = SentimentDatasetBOW("data/train.txt")
    dev_data = SentimentDatasetBOW("data/dev.txt")
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data loaded in : {elapsed_time} seconds")

    start_time = time.time()

    def plot_graph(nn2_train_accuracy, nn3_train_accuracy, nn2_test_accuracy, nn3_test_accuracy, name_pref):
        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'plots/'+name_pref+'_train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'plots/'+ name_pref + '_dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")


    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        plot_graph(nn2_train_accuracy, nn3_train_accuracy, nn2_test_accuracy, nn3_test_accuracy, 'BOW')
        
        # plt.show()

    elif args.model == "DAN":
        print('\n2 layers for the pretrained intiialisations:')

        emb_file = 'data/glove.6B.'+args.glove_size+'d-relativized.txt'
        train_data = SentimentDatasetDAN("data/train.txt", emb_file)
        dev_data = SentimentDatasetDAN("data/dev.txt", emb_file)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)
        
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2DAN(input_size=int(args.glove_size), hidden_size=100), train_loader, test_loader)
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3DAN(input_size=int(args.glove_size), hidden_size=100), train_loader, test_loader)

        plot_graph(nn2_train_accuracy, nn3_train_accuracy, nn2_test_accuracy, nn3_test_accuracy, 'DAN')


    
        print('\n2 layers for the randomised initialisations:')
        train_data = SentimentDatasetRandomDAN("data/train.txt", emb_file)
        dev_data = SentimentDatasetRandomDAN("data/dev.txt", emb_file)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn = train_data.collate_fn)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False, collate_fn = dev_data.collate_fn)

        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2RandomDAN(input_size=train_data.input_size, hidden_size=300), train_loader, test_loader)
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3RandomDAN(input_size=train_data.input_size, hidden_size=300), train_loader, test_loader)

        plot_graph(nn2_train_accuracy, nn3_train_accuracy, nn2_test_accuracy, nn3_test_accuracy, 'RandomDAN')

    elif args.model == "SUBWORDDAN":
        
        emb_file = 'data/glove.6B.'+args.glove_size+'d-relativized.txt'
        f = open(emb_file)

        cnt_dictionary = {}

        for line in f:
            if line.strip() != "":
                
                space_idx = line.find(' ')
                word = line[:space_idx]
                if word not in cnt_dictionary:
                    cnt_dictionary[word] = 1
                else:
                    cnt_dictionary[word] += 1

        def get_duet_frequencies(corpus):
            duets = defaultdict(int)
            for word, freq in corpus.items():
                chars = word.split()
                for i in range(len(chars) - 1):
                    duets[(chars[i], chars[i + 1])] += freq
            return duets

        def merge_duet(duet, corpus):
            new_corpus = {}
            cont_chars = ' '.join(duet)
            replacement = ''.join(duet)
            for word, freq in corpus.items():
                new_word = word.replace(cont_chars, replacement)
                new_corpus[new_word] = freq
            return new_corpus

        def bpe(corpus, num_merges):
            corpus = {' '.join(list(word)): freq for word, freq in corpus.items()}
            vocab = set(char for word in corpus for char in word.split())

            for _ in range(num_merges):
                duets = get_duet_frequencies(corpus)
                if not duets:
                    break
                
                best_duet = max(duets, key=duets.get)
                vocab.add(''.join(best_duet))
                corpus = merge_duet(best_duet, corpus)
            
            return vocab, corpus

        corpus = cnt_dictionary
        merge_num = 10000
        vocab, final_bpe_corpus = bpe(corpus, merge_num)

        train_data = SentimentDatasetRandomBPE("data/train.txt", emb_file, vocab) 
        dev_data = SentimentDatasetRandomBPE("data/dev.txt", emb_file, vocab)

        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn = train_data.collate_fn)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False, collate_fn = dev_data.collate_fn)

        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2RandomBPE(input_size=train_data.input_size, hidden_size=300), train_loader, test_loader)
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3RandomBPE(input_size=train_data.input_size, hidden_size=300), train_loader, test_loader)

        plot_graph(nn2_train_accuracy, nn3_train_accuracy, nn2_test_accuracy, nn3_test_accuracy, 'BPE')

        f.close()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Trained Models in : {elapsed_time} seconds")
        

if __name__ == "__main__":
    main()

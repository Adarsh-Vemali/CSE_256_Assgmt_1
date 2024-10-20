You can run the entire code with 2 commands as mentioned in the assignment.

python main.py --model DAN
python main.py --model SUBWORDDAN

Files and folders created:
1. DANmodels.py: 
    SentimentDatasetDAN:
        Helps in creating the dataset for the neural network that we will use later. Here within the init function we calculate the embeddings from the GLOVE embeddings(300d) and then we use these embeddings for all words in the dataset and then average them and these are used as the dataset(doing this enables us to work without padding which is however done for the rest of the parts). This dataset is then passed throught the defined neural network with a hidden size 300. 
    SentimentDatasetRandomDAN:
        Creates a dataset that enables the random DAN. Here we use the word indexer to store the indices of the different words in a list. This is then padded within each batch using the collate_fn(custom defined). So now we have a tensor of tensors which is padded till the longest batch sentence length, I have ensured that I ignore the padded values. Ths is again passed through neural networks (2 and 3 layer) with simple hidden_sizes of 300. These indices are then used by putting through the embedding layer(trainable) and then we pass it through the nerual net.
    SentimentDatasetRandomBPE:
        break_sentences_into_subwords function converts sentences into the subwords that are defined by the vocabulary built using the BPE methodology. Rest of the code is very similar to the SentimentDatasetRandomDAN, as here also we use randomised intiialisations for our embeddings.

2. plots:
    This is where all our graphs are stored. The prefix tells you about which setting the graph was made by. Further we have both the dev and train accuracies for every epoch that the neural net was trained for.

Files edited:
1. sentiment_data.py
    read_word_embeddings_random: function helps in creating the word indexer for the randomised vectors in RandomDAN.
    read_word_embeddings_random_BPE: function helps in creating the word indexer for the randomised vectors in the SUBWORD setting
2. main.py
    I have made changes to facilitate running of the code for models like DAN and SUBWORDDAN

Please note that the code is for using 'glove.6B.300d-relativized.txt' as the default if you wish to run the code for 50dimensional glove embeddings run:

python main.py --model DAN glove_size = 50

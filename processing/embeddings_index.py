from collections import defaultdict

from data.CustomDataset import CustomDataset

glove_dir = '../glove/glove.6B.200d.txt'
embeddings_index = {} # empty dictionary
import numpy as np
# Load the dataset
dataset = CustomDataset('../data/captions.txt')

word_counts = defaultdict(int)  # A defaultdict to count word appearances
max_length = 0

for i in range(len(dataset)):
    _, caption = dataset[i]

    words = caption.split()
    max_length = max(max_length, len(words))  # Update max_length
    for w in words:
        word_counts[w] += 1  # Increment the count for the word

print(len(word_counts))  # Output: Total unique words
print(max_length)  # Output: Maximum caption length

# Filter words that appear more than the threshold
word_count_threshold = 10
vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('After preprocessed %d -> %d' % (len(word_counts), len(vocab)))



i2w = {}
w2i = {}

id = 1
for w in vocab:
    w2i[w] = id
    i2w[id] = w
    id += 1

print(len(i2w), len(w2i))
print(i2w[300])


embedding_dim = 200
vocab_size = len(vocab) + 1

with open(glove_dir , encoding="utf-8") as file:
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    file.close()
    print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 200
vocab_size = len(vocab) + 1

embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in w2i.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector
print(embedding_matrix.shape)

from pickle import dump, load
with open("embedding_matrix.pkl", "wb") as file:
    dump(embedding_matrix, file)
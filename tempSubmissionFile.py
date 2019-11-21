import time
import numpy as np
import pandas as pd
from pandas_ml import ConfusionMatrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple
import sys


class Tokenizer():
    def __init__(self, threshold=20):
        self.word2int = {}
        self.threshold = threshold
        self.word_counts = {}

    def _count_words(self, text):
        for sentence in text:
            for word in sentence.split():
                if word not in self.word_counts:
                    self.word_counts[word] = 1
                else:
                    self.word_counts[word] += 1
        print("Size of Vocabulary: ", len(self.word_counts))

    def fit_on_texts(self, texts, embeddings_index):
        self._count_words(texts)
        token_index = 0
        for word, count in self.word_counts.items():
            if count >= self.threshold or word in embeddings_index:
                self.word2int[word] = token_index
                token_index += 1
        special_characters = ["<unk>", "<pad>"]
        for c in special_characters:
            self.word2int[c] = len(self.word2int)

        usage_ratio = round(len(self.word2int) / len(self.word_counts), 4) * 100
        print("Total number of unique words:", len(self.word_counts))
        print("Number of words we will use:", len(self.word2int))
        print("Percent of words we will use: {}%".format(usage_ratio))

    def text_to_sequence(self, text, pred=False):
        if pred:
            seq = []
            for word in text.split():
                if word in self.word2int:
                    seq.append(self.word2int[word])
                else:
                    seq.append(self.word2int["<unk>"])
            return seq
        else:
            seq = []
            for s in text:
                temp_seq = []
                for word in s.split():
                    if word in self.word2int:
                        temp_seq.append(self.word2int[word])
                    else:
                        temp_seq.append(self.word2int["<unk>"])
                seq.append(temp_seq)
            return seq


task = str(sys.argv[1])
embeddingPath = "numberbatch-en.txt"
preprocessedDatafile = "processedFile.csv"

preprocessedData = pd.read_csv(preprocessedDatafile)

hyperparameters = {
    'layersCnt': 2,
    'batchSize': 64,
    'epochs': 5,
    'hiddenUnits': 64,
    'dropoutProb': 0.8,
    'dimEmbeddings': 300,
    'validationData': 0.2,
    'learningRateDecay': 0.95,
    'learningRate': 0.005,
    'checkUpdatesize': 500
}

NUM_CLASSES = 6


def load_embeddings(path='./embeddings/numberbatch-en.txt'):
    embeddings_index = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding
    return embeddings_index


def create_embedding_matrix(word2int, embeddings_index, embedding_dim=300):
    nb_words = len(word2int)
    word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
    for word, i in word2int.items():
        if word in embeddings_index:
            word_embedding_matrix[i] = embeddings_index[word]
        else:
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            embeddings_index[word] = new_embedding
            word_embedding_matrix[i] = new_embedding
    print("Length Embedding Matrix: %d\tLength Word2Int: %d" % (len(word_embedding_matrix), len(word2int)))
    return word_embedding_matrix

def pad_batch(batch, word2int):
    lengths = []
    for text in batch:
        lengths.append(len(text))
    max_length = max(lengths)
    pad_text = tf.keras.preprocessing.sequence.pad_sequences(batch,
                                                             maxlen=max_length,
                                                             padding='post',
                                                             value=word2int['<pad>'])
    return pad_text


def get_batches(x, y, batch_size, word2int):
    for batch_i in range(0, len(x)//batch_size):
        start = batch_i * batch_size
        end = start+batch_size
        batch_x = x[start:end]
        labels = y[start:end]
        pad_batch_x = np.asarray(pad_batch(batch_x, word2int))
        yield pad_batch_x, labels


def get_test_batches(x, batch_size, word2int):
    for batch_i in range(0, len(x)//batch_size):
        start = batch_i * batch_size
        end = start+batch_size
        batch = x[start:end]
        pad_batch_test = np.asarray(pad_batch(batch, word2int))
        yield pad_batch_test



def model_inputs():
    inp = tf.placeholder(tf.int32, [None, None], name='input')
    target = tf.placeholder(tf.int32, [None, None], name='labels')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    keep_probability = tf.placeholder(tf.float32, name='keep_prob')
    return inp, target, learning_rate, keep_probability


tokenizer = Tokenizer()
embeddings_index = load_embeddings(embeddingPath)
tokenizer.fit_on_texts(preprocessedDatafile.text, embeddings_index)

word_embedding_matrix = create_embedding_matrix(tokenizer.word2int, embeddings_index, hyperparameters['dimEmbeddings'])
seq = tokenizer.text_to_sequence(preprocessedDatafile['text'])

tf.reset_default_graph()
train_graph = tf.Graph()
with train_graph.as_default():
    with tf.name_scope("inputs"):
        input_data, labels, lr, keep_prob = model_inputs()
        weight = tf.Variable(
            tf.truncated_normal([hyperparameters['hiddenUnits'], NUM_CLASSES],
                                stddev=(1 / np.sqrt(hyperparameters['hiddenUnits'] * NUM_CLASSES))))
        bias = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))

    embeddings = word_embedding_matrix
    embs = tf.nn.embedding_lookup(embeddings, input_data)

    with tf.name_scope("RNN_Layers"):
        stacked_rnn = []
        for layer in range(hyperparameters['layersCnt']):
            cell_fw = tf.contrib.rnn.GRUCell(hyperparameters['hiddenUnits'])
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                    output_keep_prob=keep_prob)
            stacked_rnn.append(cell_fw)
        multilayer_cell = tf.contrib.rnn.MultiRNNCell(stacked_rnn, state_is_tuple=True)

    with tf.name_scope("init_state"):
        initial_state = multilayer_cell.zero_state(hyperparameters['batchSize'], tf.float32)

    with tf.name_scope("Forward_Pass"):
        output, final_state = tf.nn.dynamic_rnn(multilayer_cell,
                                                embs,
                                                dtype=tf.float32)

    with tf.name_scope("Predictions"):
        last = output[:, -1, :]
        predictions = tf.exp(tf.matmul(last, weight) + bias)
        tf.summary.histogram('predictions', predictions)

    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=labels))
        tf.summary.scalar('cost', cost)

    # Optimizer
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

    # Predictions comes out as 6 output layer, so need to "change" to one hot
    with tf.name_scope("accuracy"):
        correctPred = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    export_nodes = ['input_data', 'labels', 'keep_prob', 'lr', 'initial_state', 'final_state',
                    'accuracy', 'predictions', 'cost', 'optimizer', 'merged']

    merged = tf.summary.merge_all()

print("Graph is built.")
graph_location = "./graph"

Graph = namedtuple('train_graph', export_nodes)
local_dict = locals()
graph = Graph(*[local_dict[each] for each in export_nodes])

print(graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(train_graph)


def train(x_train, y_train, batch_size, keep_probability, learning_rate, display_step=20, update_check=500):
    print("Training Now")
    epochs = hyperparameters['epochs']
    summary_update_loss = []
    min_learning_rate = 0.0005
    stop_early = 0
    stop = 3
    checkpoint = "./saves/best_model.ckpt"
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter('./summaries' + '/train', sess.graph)

        for epoch_i in range(1, epochs + 1):
            state = sess.run(graph.initial_state)

            update_loss = 0
            batch_loss = 0

            for batch_i, (x, y) in enumerate(get_batches(x_train, y_train, batch_size, tokenizer.word2int)):
                if batch_i == 1 and epoch_i == 1:
                    print("Starting")
                feed = {graph.input_data: x,
                        graph.labels: y,
                        graph.keep_prob: keep_probability,
                        graph.initial_state: state,
                        graph.lr: learning_rate}
                start_time = time.time()
                summary, loss, acc, state, _ = sess.run([graph.merged,
                                                         graph.cost,
                                                         graph.accuracy,
                                                         graph.final_state,
                                                         graph.optimizer],
                                                        feed_dict=feed)
                if batch_i == 1 and epoch_i == 1:
                    print("Finished first")

                train_writer.add_summary(summary, epoch_i * batch_i + batch_i)

                batch_loss += loss
                update_loss += loss
                end_time = time.time()
                batch_time = end_time - start_time

                if batch_i % display_step == 0 and batch_i > 0:
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Acc: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(epoch_i,
                                  epochs,
                                  batch_i,
                                  len(x_train) // batch_size,
                                  batch_loss / display_step,
                                  acc,
                                  batch_time * display_step))
                    batch_loss = 0

                if batch_i % update_check == 0 and batch_i > 0:
                    print("Average loss for this update:", round(update_loss / update_check, 3))
                    summary_update_loss.append(update_loss)

                    # If the update loss is at a new minimum, save the model
                    if update_loss <= min(summary_update_loss):
                        print('New Record!')
                        stop_early = 0
                        saver = tf.train.Saver()
                        saver.save(sess, checkpoint)

                    else:
                        print("No Improvement.")
                        stop_early += 1
                        if stop_early == stop:
                            break
                    update_loss = 0

            learning_rate *= hyperparameters['learningRateDecay']
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            if stop_early == stop:
                print("Stopping Training.")
                break
    print("Done Training")


def test(x_test, y_test):
    print("Testing Now")
    with tf.Session(graph=train_graph) as sess:
        checkpoint = "./saves/best_model.ckpt"
        all_preds = []

        saver = tf.train.Saver()

        saver.restore(sess, checkpoint)
        state = sess.run(graph.initial_state)
        print("Total Batches: %d" % (len(x_test) // hyperparameters['batchSize']))
        for ii, x in enumerate(get_test_batches(x_test, hyperparameters['batchSize'], tokenizer.word2int), 1):
            if ii % 100 == 0:
                print("%d batches" % ii)
            feed = {graph.input_data: x,
                    graph.keep_prob: hyperparameters['dropoutProb'],
                    graph.initial_state: state}

            test_preds = sess.run(graph.predictions, feed_dict=feed)

            for i in range(len(test_preds)):
                all_preds.append(test_preds[i, :])

    all_preds = np.asarray(all_preds)
    y_predictions = np.argmax(all_preds, axis=1)
    y_true = y_test.argmax(axis=1)
    y_true = y_true[:y_predictions.shape[0]]

    cm = ConfusionMatrix(y_true, y_predictions)
    cm.plot(backend='seaborn', normalized=True)
    plt.title('Confusion Matrix Stars prediction')
    plt.figure(figsize=(12, 10))

    test_correct_pred = np.equal(y_predictions, y_true)
    test_accuracy = np.mean(test_correct_pred.astype(float))

    print("Test accuracy is: " + str(test_accuracy))


ratings = preprocessedData.stars.values.astype(int)
ratings_cat = tf.keras.utils.to_categorical(ratings)
x_train, x_test, y_train, y_test = train_test_split(seq, ratings_cat, test_size=0.2, random_state=9)

if task == 'train':
    train(x_train, y_train, hyperparameters['batchSize'], hyperparameters['dropoutProb'], hyperparameters['learningRate'], update_check=hyperparameters['checkUpdatesize'])
    test(x_test, y_test)
elif task == 'test':
    test(x_test, y_test)

import os
import pandas as pd
from DBN import DBN

'''
each sample is 6 dimensional, m=6, this is visible layer
want final hidden or latent layer to be 2 or 3 dim in order to visualize
'''

sample_freq = 40 # samples per second in data
sample_window = 4 # number of seconds in batch
batch_size = int(sample_freq * sample_window)
#data_dir = '../data/TrainingData/'
data_dir = '/Users/liam_adams/my_repos/ece765_gait_id/data/TrainingData/'

# need to normalize it so all entries are in range [0,1]
def load_training_data():
    x_batches = []
    for f in os.listdir(data_dir):
        filename = filename = os.fsdecode(f)
        if filename.endswith('x.csv'):            
            start_index = 0
            df = pd.read_csv(data_dir + filename)
            while start_index <= len(df) - batch_size:                
                x_batch = df[start_index:start_index + batch_size]
                x_batches.append(x_batch)
                start_index += int(batch_size / 2) # 50% overlap

    return x_batches


def trainDBN(learning_rate, k1, k2, epochs, batch_size, dims):
    x_train = load_training_data() # x_train should be list of 6 dim vectors
    # parse string input into integer list
    dims = [int(el) for el in dims.split(",")]
    dbn = DBN(dims, learning_rate, k1, k2, epochs, batch_size)
    dbn.train_PCD(x_train)
    # dump dbn pickle
    f = open("pickles/"+current_time+"/dbn.pickle", "wb")
    pickle.dump(dbn, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

if __name__ == "__main__":
    # dimensions is list of [visible, hidden, visible, hidden, etc]
    # k1 used in PCD, time steps for initial sample
    # k2 used in CD and PCD for all other samples
    k1 = 1
    k2 = 5
    dimensions = '6,3,6,3,6,3'
    learning_rate = .01
    epochs = 5
    trainDBN(learning_rate, k1, k2, epochs, batch_size, dimensions)
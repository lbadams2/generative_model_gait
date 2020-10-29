

'''
each sample is 6 dimensional, m=6, this is visible layer
want final hidden or latent layer to be 2 or 3 dim in order to visualize
'''

# need to normalize it so all entries are in range [0,1]
def load_training_data():
    pass

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
    k1 = 5
    dimensions = '6,3,6,3,6,3'
    trainDBN(learning_rate, k1, k2, epochs, batch_size, dimensions)
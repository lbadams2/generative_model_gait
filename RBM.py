

class RBM:

    # k1 is number of gibbs sampling steps before PCD (persistent contrastive divergence)
    # k2 is number of gibbs sampling steps during PCD
    def __init__(self, visible_neurons, hidden_neurons, learning_rate, k1, k2, epochs, batch_size):
        self.visible_neurons = visible_neurons
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.k1 = k1
        self.k2 = k2
        self.epochs = epochs
        self.batch_size = batch_size

        # initial weight matrix mapping visible to hidden layer
        self.w = tf.Variable(tf.random.uniform(shape=(num_hidden,num_visible), maxval=1, minval=-1),dtype="float32")
        # initial bias vector for hidden variables
        self.b_h = tf.Variable(tf.random.uniform(shape=(num_hidden,1), maxval=1, minval=-1),dtype="float32")
        # initial bias vector for visible variables
        self.b_v = tf.Variable(tf.random.uniform(shape=(num_visible,1), maxval=1, minval=-1),dtype="float32")

    # sample visible 
    def gibbs_sampling(v, k):
        pass
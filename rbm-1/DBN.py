

class DBN:
    # k1 used in PCD, time steps for initial sample
    # k2 used in CD and PCD for all other samples
    def __init__(self, dims, learning_rate = 0.01, k1 = 1, k2 = 5, epochs = 1, batch_size = 5):
        # create list of RBMs, dims is list even number of elems, first pair is visible/hidden for first RBM, second pair is v/h, etc
        # each RBM initialized with weights and biases
        self.models = [RBM(num_visible=dims[i],num_hidden=dims[i+1],
                           learning_rate=learning_rate, k1=k1, k2=k2, epochs=epochs,
                           batch_size=batch_size) for i in range(len(dims)-1)]
        self.top_samples = None

    def train_PCD(self, data):        
        for i in range(len(self.models)): # for each RBM
            print("Training RBM: %s" % str(i+1))
            # this will sample Markov chain, compute gradient, update params (weights, biases) using Adam
            # will do this repeatedly on the 2 visible and hidden layers for the specified number of epochs
            self.models[i].persistive_contrastive_divergence_k(data)
            if i != len(self.models)-1:
                print("Sampling data for model: %s" % str(i+2))
                # initialize next visible layer not yet trained with hidden layer of just trained RBM
                self.models[i+1].b_v = tf.Variable(self.models[i].b_h)
                # if next RBMs weight matrix is same shape as transpose of just trained RBMs weight matrix (should be?)
                if self.models[i+1].w.get_shape().as_list() == tf.transpose(self.models[i].w).get_shape().as_list():
                    print("Assigning previously learned transpose-weights to next model")
                    self.models[i+1].w = tf.Variable(tf.transpose(self.models[i].w))
                    self.models[i+1].b_h = tf.Variable(self.models[i].b_v)

                # this will make each sample in data a vector of 0s or 1s (why?)
                # first RBM gets real data, remaining RBMs get one hot
                data = [self.models[i].random_sample(self.models[i].h_given_v(img)) for img in data]
            else:
                print("Final model, no generation for next model")
                self.top_samples = data
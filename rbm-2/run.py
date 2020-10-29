

# need to normalize it so all entries are in range [0,1]
def load_training_data():
    pass

if __name__ == '__main__':
    rbm = RBM_Impl()
    X = load_training_data()
    rbm.fit(X)
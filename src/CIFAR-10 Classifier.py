#%% Importing Dependencies
import numpy as np
import time
import math
#%% Helper Functions
# =============================================================================
# HELPER FUNCTIONS FOR LOADING DATA
# =============================================================================

# for unpickling the CIFAR-10 datasets
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Pre-process and load the data
def DataLoad(num_of_batches):
    X, y = LoadBatches(num_of_batches)
    X = normalize(X)
    X = X.reshape(10000*num_of_batches, 3072)
    y = oneHot_encoder(y)
    y = y.reshape(-1, 10)
    y = y.astype('uint8')
    return X, y

#Load the required number of batches
def LoadBatches(num_of_batches):
    for i in range(0,num_of_batches):
        filename = "cifar-10-batches-py/data_batch_" + str(i+1)
        data = unpickle(filename)
        if i == 0: #for first read make X and y 
            X = data[b"data"]
            y = np.array(data[b"labels"])
        else: # for the rest of the reads append to previously readed X,y
            X = np.append(X, data[b"data"], axis=0)
            y = np.append(y, data[b"labels"], axis=0)

    return X, y

# one hot encoded labels
def oneHot_encoder(dataIn):
    oneHot_vector = np.zeros((dataIn.shape[0], 10))
    oneHot_vector[np.arange(dataIn.shape[0]), dataIn] = 1
    return oneHot_vector

def normalize(dataIn):
    return dataIn/255.

# split data into test data and train data
def test_train_split(X, y, split_ratio=0.8):
    
    '''
    X,y: 
        data to be splitted
        Shapes: (number of training examples, number of features),
                (number of training examples, number of classes) respectively
    '''
    split = int(split_ratio * X.shape[0])
    idxs = np.random.permutation(X.shape[0]) #array of randome indexes
    training_idx, val_idx = idxs[:split], idxs[split:]
    # split the data by the calculated indexes
    X_train, X_val = X[training_idx, :], X[val_idx, :]
    y_train, y_val = y[training_idx, :], y[val_idx, :]



    return X_train, y_train, X_val, y_val


#%% Neural Network Class and functions
    
# =============================================================================
# HELPER FUNCTIONS FOR NEURAL NETWORK CLASS 
# =============================================================================

def NonLinearity(out,activation):
    '''
    `out`: 
        Z (scores)
    '''

    if activation == 'logistic':
        return 1.0 / (1.0 + np.exp(-out))

    elif activation == 'softmax':
        expnnt=np.exp(out)
        z=expnnt/np.sum(expnnt,axis=0)
        return z

    elif activation == 'relu':
        return out*(out>0)


def delta_NonLinearity(out,activation='logistic'):
    '''
    `out`: 
        Z (scores)
    '''
    
    if activation == 'logistic':
        return NonLinearity(out,activation) * (1 - NonLinearity(out,activation))
    elif activation == 'relu':
        return (out>0)+0


def SigmoidLoss(actvn, y_act):
    '''
    `actvn`: 
        output probabilites of neural network
    `y_act`:
        actual class labels
    '''
    
    m = len(y_act.T)
    return (np.sum(np.nan_to_num(
            y_act*np.log(actvn)+(1-y_act)*np.log(1-actvn ))))/-m

def SoftmaxLoss(a, y):
    '''
    `a`: 
        output probabilites of neural network
    `y`:
        actual class labels
    '''
    m = y.shape[1]
    z=np.sum(np.multiply(y,np.log(a)),axis=0)
    z=z[np.newaxis,:]
    L = (-1/m)*np.sum(z)
    return L



# =============================================================================
# NEURAL NETWORK CLASS
# =============================================================================
class NeuralNetwork(object):

    def __init__(self, layers_sizes, verbose = True,
                 hidden_layers_activation = 'relu', reg = 0.01, 
                 solver = 'sgd', momentum = 0.9):
        
        '''
        NeuralNetwork V.0.1
        This neural network object can support multiple layers
        Current version is designed to support the classification of CIFAR-10 
        dataset. 
        supported non_liniearites: 
            relU, sigmoid.
        supported solvers:
            sgd, momentum
        '''
        
        #apply some
        self.momentum = momentum
        self.solver = solver

        self.regularization = reg
        self.nonLinearity = hidden_layers_activation
        self.verbosity = verbose
        self.num_of_layers = len(layers_sizes)

        l_nums = np.arange(len(layers_sizes[:-1]))

        self.w_keys = ['W'+str(i) for i in l_nums]
        self.b_keys = ['b'+str(i) for i in l_nums]
        b_init_ = [np.random.randn(n_rows, 1) for n_rows in layers_sizes[1:]]
        w_init_ = [np.random.randn(n_rows, n_cols)/np.sqrt(n_cols)
                            for n_cols, n_rows in zip(layers_sizes[:-1],
                                                      layers_sizes[1:])]
        vx_b_init_ = [np.zeros((n_rows, 1)) for n_rows in layers_sizes[1:]]
        vx_w_init_ = [np.zeros((n_rows, n_cols))/np.sqrt(n_cols)
                            for n_cols, n_rows in zip(layers_sizes[:-1],
                                                      layers_sizes[1:])]
        vW = dict(zip(self.w_keys,vx_w_init_))
        vb = dict(zip(self.b_keys,vx_b_init_))
        W = dict(zip(self.w_keys,w_init_))
        b = dict(zip(self.b_keys,b_init_))
        self.params =  {**W, **b}
        self.V =  {**vW, **vb}
        self.loss = 0
        self.loss_history = []
        self.iterations = 0

    def forward_pass(self, x):
        
        layer = x
        layers_activations = []  # list to store activations for every layer
        layers_scores = []  # list to store scores for every layer
        
        len_keys = self.num_of_layers - 1 # How many number of layers
        i = 0
        for b_key, w_key in zip(self.b_keys, self.w_keys): #for each layer
            #get the layer scores
            score = np.dot(self.params[w_key], layer) + self.params[b_key]
            layers_scores.append(score)
            
            #if the layer is last layer then apply softmax
            if i == (len_keys-1):
                layer = NonLinearity(score, activation ='softmax')
            else: # apply hidden layer nonlinearity on rest of hidden layers
                layer = NonLinearity(score,activation = self.nonLinearity)
            layers_activations.append(layer)
            i += 1
        # store the calculated scores and activations
        self. activations = layers_activations
        self.scores = layers_scores
        return layers_scores, layers_activations

    def squared_parameter_norm(self):
        '''
        required for calculating regularization cost
        '''
        W = self.params
        norm_W = 0;
        for wkey in W.keys():
            norm_W += np.sum(W[wkey]*W[wkey])
        return norm_W


    def compute_loss(self,yhat,y_act):
        #get the softmax cost from the softmax cost function
        cost = SoftmaxLoss(yhat,y_act)
        # calculate regularization cost
        m = yhat.shape[1]
        param_norm = self.squared_parameter_norm()
        reg_loss = 0.5*self.regularization*param_norm*(1/m)
        # add the cost to the regularization cost
        loss=cost + reg_loss
        return loss

    def backward_pass(self, X, y):
        Z,A = self.forward_pass(X)
        cost = self.compute_loss(A[-1],y)

        dWs = {}
        dbs = {}

        n_layer = self.num_of_layers-2
        m = y.shape[1]
        # delta for output layer
        dZ = (A[-1] - y)/m
        
        # calculate gradients for hidden layer
        for wkey,bkey in zip(self.w_keys[:0:-1], self.b_keys[:0:-1]):

            dW_key = 'dW'+str(n_layer)
            db_key = 'db'+str(n_layer)

            dWs[dW_key] =  dZ.dot(A[n_layer-1].T)/m
            dbs[db_key] =  np.sum(dZ, axis=1)/m

            dA = np.dot(self.params[wkey].T,dZ)
            dZ= np.multiply(dA,delta_NonLinearity(Z[n_layer-1],
                                                  activation = self.nonLinearity))

            n_layer -= 1

        # calculate gradients for input layer
        wkey = self.w_keys[0]
        dW_key = 'dW'+str(0)
        db_key = 'db'+str(0)

        dWs[dW_key] =  dZ.dot(X.T)/m
        dbs[db_key] =  np.sum(dZ, axis=1)/m

        self.dW = dWs
        self.db = dbs
        return dWs, dbs, cost

    def get_next_batch(self, X, y, mini_batch_size):
        '''
        generator object to get batch of the required size. this method does
        not cunsume much memory and yeilds a mini batch whenever generator 
        object is called in the next() function
        '''
        for idx in range(0, X.shape[1], mini_batch_size):
            idx1 = idx + mini_batch_size
            mini_batch = [X[:,idx:idx1],y[:,idx:idx1]]
            yield mini_batch
    
    def parameter_update(self,W,b,dW,db,lr,epsilon,wkey,bkey):
        
        if self.solver == 'sgd': # simple parameter update for gradient descent
            W = W*epsilon -lr*dW
            b += -lr*db.reshape(-1,1)
            return W,b
        elif self.solver == 'momentum': # when solver is momentum
            self.V[wkey] = self.momentum * self.V[wkey] + dW
            self.V[bkey] = self.momentum * self.V[bkey] + db.reshape(-1,1)
            W = W*epsilon -lr*self.V[wkey]
            b += -lr*self.V[bkey]
            return W,b
                
            

    def train(self, X, y, batch_size=200, learning_rate=1, epochs=1000):

        # determing number of mini batches to loop through
        num_of_mini_batches = y.shape[1] / batch_size
        num_of_mini_batches = int(np.ceil(num_of_mini_batches))
        
        n_epoch = 1 # set current epoch to 1 (for counting purposes)
        c = epochs*num_of_mini_batches
        for j in range(epochs): # for each epoch
            nbatch = 1
            # get new mini batch generator
            batch_iterable = self.get_next_batch(X, y, batch_size)
            for i in range(num_of_mini_batches): # for each mini batch
                btime = time.time() # for time calculation
                
                # get the mini batch values
                mini_batch = next(batch_iterable)
                mini_X = mini_batch[0]
                mini_y = mini_batch[1]
                
                # get gradients from bach probagation for
                dW,dB,cost = self.backward_pass(mini_X,mini_y)
                for i in range(self.num_of_layers-1): # for layer
                    
                    # get the keys to access gradients and parameters for each 
                    # layer
                    dwkey = 'dW'+str(i)
                    dbkey = 'db'+str(i)
                    wkey = 'W'+str(i)
                    bkey = 'b'+str(i)
                    m = mini_y.shape[1]
                    dw = dW[dwkey] 
                    db = dB[dbkey]
                    
                    # get the parameters for the current layer
                    W = self.params[wkey].copy()
                    b = self.params[bkey].copy()
                    
                    # epsilon used for regularization
                    epsilon = (1 - learning_rate*self.regularization/m )
#
                    # get updated parameters
                    W,b = self.parameter_update(W,b,dw,db,
                                                learning_rate,epsilon,
                                                wkey,bkey)
                    # store the updated paramters
                    self.params[wkey] = W
                    self.params[bkey] = b
                
                #for ETA measurement
                etime = time.time() - btime
                c = c-1
                eta = etime*c
                hours = math.floor(eta/3600)
                eta = eta - hours*3600
                mins = math.floor(eta/60)
                eta = eta - mins*60
                secs = round(eta%60)
#                print(' ETA:',hours,'h :',mins,'m :',secs,'s')
                self.loss_history.append(cost)
                self.iterations += 1
                # print the information
                print('iteration: %d , batch:(%d/%d) , epoch:(%d/%d), cost: %f, ETA: %dh:%dm:%ds '
                      %(self.iterations, nbatch,num_of_mini_batches,n_epoch, epochs, cost,
                        hours, mins, secs) )
                nbatch += 1 # update batch counter

            n_epoch += 1 # update epoch counter

    def Evaluate(self, X_eval, y_eval):
        '''
        Parameters:
            `X_eval` is matrix of examples. 
            Shape: (number of examples, number of features)
            `y_eval` is matrix of examples. 
            Shape: (number of examples, number of classes)
        Output:
            a scaler of containing accuracy measure.  
            Shape: ()
        '''

        ypred = model.predict(X_eval)
        acc = y_eval.shape[0] # asssume 100% accuracy by setting accuracy counter to 10000
        for i in range(0, acc): # loop through each test and predicted label
            if (ypred[i] != y_eval[i]).any(): # if the two oneHot vectors are not equal
                acc = acc-1                   # then reduce the true_positive counter
        accuracy = acc/y_eval.shape[0] * 100 # calculate accuracy in percentage
        return accuracy

    def predict_proba(self, X):
        
        '''
        Parameters:
            `X` is matrix of training examples. 
            Shape: (number of examples, number of features)
        Output:
            A matrix of column vectors containig prediction probablities.  
            Shape: (number of examples, classes)
        '''
        X = X.T
        s,a = self.forward_pass(X)
        preds = a[-1]
        return preds.T

    def predict(self, X):
        '''
        Parameters:
            `X` is matrix of training examples. 
            Shape: (number of examples, number of features)
        Output:
            A matrix of one-hot column vectors.  
            Shape: (number of examples, classes)
        '''
        X = X.T
        s,a = self.forward_pass(X)
        ypobs = a[-1]
        yidx = np.argmax(ypobs,axis=0).reshape(1,-1)
        ypred = np.zeros(ypobs.shape)
        ypred[    yidx , np.arange( ypobs.shape[1])  ] = 1
        return ypred.T



#%% 
# =============================================================================
# lOAD DATA INTO THE VARIABLE WORKSPACE
# =============================================================================
X, y = DataLoad(num_of_batches=5)
X_train, y_train, X_val, y_val = test_train_split(X, y, split_ratio = 0.8 )
del X,y # CLEARING DATA AFTER USE

#%%
# =============================================================================
# MAKE A NEURAL NETWORK OBJECT AND START THE TRAINING
# =============================================================================
model = NeuralNetwork([3072, 1024, 64, 10], solver='momentum', reg = 0.1, 
                      momentum = 0.99)

model.train(X_train.T,y_train.T,epochs = 25,batch_size=500)


#%%
#--make a noise when training is completed--
import winsound
winsound.Beep(500, 250)
winsound.Beep(700,250)
winsound.Beep(900, 250)
winsound.Beep(750,250)
#%%
# =============================================================================
# SHOW LOSS FUNCTION CURVE
# =============================================================================
import matplotlib.pyplot as plt
plt.plot(model.loss_history)
plt.xlabel('Number of Iterations')
plt.ylabel('Loss')
plt.title('Loss Vs. Iterations')
plt.show()


#%%
# =============================================================================
# PRINT VALIDATION ACCURACY
# =============================================================================

val_accuracy = model.Evaluate(X_val,y_val)
print('Validation Accuracy = %.2f'%val_accuracy)

#%%
# =============================================================================
# ACQUIIRE TEST ACCURACY
# =============================================================================
def print_test_accuracy():
    X_test = unpickle("cifar-10-batches-py/test_batch")[b"data"] / 255.0
    X_test = X_test.reshape(-1, 3072)
    y_test = np.array(unpickle("cifar-10-batches-py/test_batch")[b"labels"])
    y_test = oneHot_encoder(y_test)
    y_test = y_test.reshape(-1, 10)
    y_test = y_test.astype('uint8')
    
    test_accuracy = model.Evaluate(X_test,y_test)
    print('\n Test accuracy is %.2f'%test_accuracy,'% \n')
#%%
# =============================================================================
# PREDICTION FUNCTION 
# =============================================================================
def prediction(Xin):
    '''
    parameters:
        `Xin` a matrix of features and number of examples.
            shape: (number of examples, number of features)
            
            Caution: the shape (n,) is not supported for single example. for
            single example make sure shape is (1,n) where n is # of features.
                hint: use a.reshape(1,-1)) for single example 
    Output:
        One hot vector of predicted labels. 
        Shape: (number of examples, number of classes)
        
                
    '''
    return model.predict(Xin)

#%%
print('\nCongratulations. You have reached the end of the program')
response = input('Do you want to print the test accuracy? [Y/N]: ')
if response == 'Y' or response == 'y':
    print_test_accuracy()
print('Use the function  prediction(Xin)  to make predictions ')
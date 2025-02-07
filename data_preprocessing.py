import numpy as np
from sklearn.model_selection import train_test_split
from data_augmentation import choose_extra_data, data_augment


#Load Labled Dataset
Xtrain = np.load('data\Xtrain1.npy')
Ytrain = np.load('data\Ytrain1.npy')
Xtest = np.load("data\Xtest1.npy")

#Load Predicted unlabled dataset
Xtrain_extra = np.load('data\Xtrain1_extra.npy')
Ytrain_extra = np.load('data\ypred_extra2.npy')

#Labled dataset reshaping
N_x, d_x = Xtest.shape
X_test_final = Xtest.reshape(N_x, 48, 48)
Xtrain1 = Xtrain.reshape(N_x, 48, 48)
N_x, d_x = Xtrain.shape
N_y = Ytrain.shape

Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, test_size=0.1, random_state=42)
Xval = Xval / 255

#Unlabled dataset reshaping
Ytrain_extra = Ytrain_extra.reshape(-1)
N_x_extra, d_x_extra = Xtrain_extra.shape
Xtrain_extra = Xtrain_extra.reshape(N_x_extra, 48, 48)

Ytrain_extra, Xtrain_extra = choose_extra_data(Ytrain_extra, Xtrain_extra)
Xtrain = np.concatenate((Xtrain, Xtrain_extra), axis=0)
Ytrain = np.concatenate((Ytrain, Ytrain_extra), axis=0)

Xtrain, Ytrain = data_augment(Xtrain, Ytrain)

# normalize
Xtrain = Xtrain / 255
Xtrain_extra = Xtrain_extra/255

# extra dataset added
Xtrain = np.concatenate((Xtrain, Xtrain_extra), axis=0)
Ytrain = np.concatenate((Ytrain, Ytrain_extra), axis=0)

np.save("results\Xtrain.npy",Xtrain)
np.save("results\Ytrain.npy", Ytrain)
np.save("results\Xval.npy", Xval)
np.save("results\Yval.npy", Yval)
np.save("results\Xtest.npy", Xtest)
import tensorflow as tf
import numpy as np

def inferense_data(model='model.h5', datapath='data\Xtrain1_extra.npy', decicion_threshold=0.001):
    model = tf.keras.models.load_model('model.h5')
    Xtrain_extra = np.load(datapath)

    #Reshape data
    N_x_extra, d_x_extra = Xtrain_extra.shape
    Xtrain_extra = Xtrain_extra.reshape(N_x_extra, 48, 48)
    Xtrain_extra = Xtrain_extra/255

    y_pred_extra = model.predict(Xtrain_extra)

    #Only get the predictions for the minority class, no-craters (0)
    y_pred_extra = (y_pred_extra > decicion_threshold).astype(int)

    no_craters = np.where(y_pred_extra == 0)[0]

    no_crater_images = Xtrain_extra[no_craters]
    print(no_crater_images.shape)
    print(no_craters.shape)
    np.save('data\Xtrain_extra_no_craters.npy', no_crater_images)
    np.save('data\y_pred_extra_no_craters.npy', y_pred_extra[no_craters])









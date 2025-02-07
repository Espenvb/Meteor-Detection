import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

def choose_extra_data(y_pred, X):
    # choose data where confidence is high
    indx = np.where((y_pred < 0.01) | (y_pred > 0.09))[0]
    x_train = X[indx]
    y_train = y_pred[indx]
    y_train = (y_train > 0.5).astype(int) # format as binary labels
    return y_train, x_train

#Use recall and precision to find the F1score
def data_augment(Xtrain1, Ytrain1):

    minority_class = 0
    majority_class = 1

    X_minority = Xtrain1[Ytrain1 == minority_class]
    X_majority = Xtrain1[Ytrain1 == majority_class]

    augmented_images = []
    augmented_labels = []

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,       # Random rotation
        width_shift_range=0.2,   # Random horizontal shift
        height_shift_range=0.2,  # Random vertical shift
        shear_range=0.2,         # Shear intensity
        zoom_range=0.2,          # Random zoom
        horizontal_flip=True,    # Random horizontal flip
        vertical_flip=True,     # Random vertical flip
        fill_mode='reflect'
    )

    for i in range(X_minority.shape[0]):
        # Reshape the image if necessary (for grayscale, this might be needed)
        image = np.expand_dims(X_minority[i], axis=-1)
        image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 48, 48, 1)
        
        # Generate batches of augmented images
        aug_iter = datagen.flow(image, batch_size=1)
        
        for git_ in range(8):  # Create 8 augmented images per original image
            augmented_image = next(aug_iter)[0].astype(np.uint8)
            augmented_images.append(augmented_image)
            augmented_labels.append(minority_class)

    for i in range(X_majority.shape[0]):
        # Reshape the image if necessary (for grayscale, this might be needed)
        image = np.expand_dims(X_majority[i], axis=-1)
        image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 48, 48, 1)
        
        # Generate batches of augmented images
        aug_iter = datagen.flow(image, batch_size=1)
        
        for _ in range(4):  # Create 4 augmented images per original image
            augmented_image = next(aug_iter)[0].astype(np.uint8)
            augmented_images.append(augmented_image)
            augmented_labels.append(majority_class)




    # Remove grayscale channel dimension
    augmented_images = np.squeeze(augmented_images, axis=-1)

    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    combined_images = np.concatenate((Xtrain1, augmented_images), axis=0)
    combined_labels = np.concatenate((Ytrain1, augmented_labels), axis=0)

    combined_images, combined_labels = shuffle(combined_images, combined_labels, random_state=42)

    y_0 = combined_labels[combined_labels == 0] # To check the augmented dataset is balanced
    y_1 = combined_labels[combined_labels == 1]
    print("y_0: ", y_0.shape)
    print("y_1: ", y_1.shape)

    return combined_images, combined_labels

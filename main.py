import numpy as np
from keras import layers, Sequential
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

Xtrain = np.load('results\combined_images.npy.npy')
Ytrain = np.load('results\combined_labels.npy.npy')
Xval = np.load('results\Xval.npy')
Yval = np.load('results\Yval.npy')
Xtest = np.load('results\Xtest.npy')

model = Sequential([
    layers.Conv2D(16, kernel_size=(3,3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Use 'softmax' for multiple classes
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy',  
            metrics=['accuracy'])

model.fit(Xtrain, Ytrain, epochs=6, batch_size=16, validation_data=(Xval, Yval))

test_loss, test_accuracy = model.evaluate(Xval, Yval)
y_pred = model.predict(Xval)
y_pred_final = (y_pred > 0.5).astype(int)

print(f'Test accuracy: {test_accuracy}')
print(f'Test loss: {test_loss}')

f1 = f1_score(Yval, y_pred_final, average='macro') # f1 score around 0.86-0.87
print(f"F1 Score: {f1:.2f}")

#Find confusion matrix
conf_mat = confusion_matrix(Yval, y_pred.round())
print(conf_mat)

k = 0
y_pred_handin = model.predict(Xtest)
y_pred_handin = (y_pred > 0.5).astype(int)

y_crater = y_pred_handin[y_pred_handin == 1]
y_no_crater = y_pred_handin[y_pred_handin  == 0]

print("y: ", y_pred_handin.shape)
print("y_crater: ", y_crater.shape)
print("y_no_crater: ", y_no_crater.shape)

y_pred_handin = y_pred_handin.round()
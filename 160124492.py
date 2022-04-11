### Aluno: Hiago dos Santos Rabelo - 160124492 ###

!pip install split-folders

from cgi import test
from tabnanny import verbose
from keras.applications.resnet import ResNet50
from keras.models import Model
from keras.layers import Input, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image_dataset import image_dataset_from_directory
import splitfolders
from sklearn.metrics import classification_report
from keras import backend as K
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import zipfile
import numpy as np
import tensorflow as tf


DATASET_REPO = 'raw_dataset'

# IMPORTANT: It's really important to upload the zip dataset to content folder into Google Colab
# The dataset must have the following name: "raw_dataset.zip" as showed in the function bellow
def download_dataset_from_drive():
    with zipfile.ZipFile("raw_dataset.zip","r") as zip_ref:
      zip_ref.extractall(".")
    # os.rename(src, dst)

def get_model():
    # based on 
    # https://github.com/hiagop22/Identify_text_non_text/blob/dev_hiago/text_non_text.py
    # from line 94 to 116
    cnn_model = ResNet50(weights='imagenet', 
                         input_shape=(224,224,3),
                         include_top=False)

    x = Flatten(name='Flatten')(cnn_model.output)
    x = Dense(1000, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(8, activation='softmax')(x)

    model = Model(inputs=cnn_model.input, outputs=x)

    for layer in cnn_model.layers:
        layer.trainable = False

    opt = RMSprop(learning_rate=0.01, )
    # I used 'sparse_categorical_crossentropy' cause the outputs from my
    # dataset arenot one_hot vectors, but integers, according to the bug descripted in 
    # this site https://python.tutorialink.com/i-created-a-cifar10-dataset-learning-model-using-a-cnn-model-why-is-there-an-error/
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])
    # model.summary()
    return model

def get_data():
    # splitfolder is based on 
    # https://stackoverflow.com/questions/53074712/how-to-split-folder-of-images-into-test-training-validation-sets-with-stratified
    # cause the raw datataset downloaded isn't already splitted
    splitfolders.ratio(DATASET_REPO, output="processed_dataset", seed=1, ratio=(0.8, 0.1, 0.1))

    train_ds = image_dataset_from_directory(directory='processed_dataset/train',
                                            image_size=(224,224),
                                            labels='inferred',
                                            batch_size=4,
                                            seed=1,
                                            shuffle=True
                                        )

    val_ds = image_dataset_from_directory(directory='processed_dataset/val',
                                            image_size=(224,224),
                                            labels='inferred',
                                            batch_size=4,
                                            seed=1,
                                            shuffle=False
                                        )

    test_ds = image_dataset_from_directory(directory='processed_dataset/test',
                                            image_size=(224,224),
                                            labels='inferred',
                                            batch_size=4,
                                            seed=1,
                                            shuffle=False
                                        )
    return train_ds, val_ds, test_ds

# Callbacks are based on:
# ModelCheckpoint: https://github.com/hiagop22/Identify_text_non_text/blob/dev_hiago/text_non_text.py
# EarlyStopping: https://keras.io/api/callbacks/early_stopping/
# ReduceLROnPlateau: https://keras.io/api/callbacks/reduce_lr_on_plateau/#:~:text=ReduceLROnPlateau%20class&text=Reduce%20learning%20rate%20when%20a,the%20learning%20rate%20is%20reduced.
callbacks = [
    ModelCheckpoint(filepath='best-model', 
                    monitor='val_loss', 
                    save_best_only=True,
                    mode='min'),
    EarlyStopping(monitor='val_loss',
                  min_delta=0.001,
                  patience=10,
                  mode='min'),
    ReduceLROnPlateau(monitor='val_loss',
                      factor=0.1,
                      patience=3,
                      min_delta=0.001,
                      min_lr=1e-6,
                      verbose=0
                      )
]

if __name__ == '__main__':
    model = get_model()
    download_dataset_from_drive()
    train_ds, val_ds, test_ds = get_data()
    model.fit(train_ds, validation_data=val_ds, epochs=200, verbose=1, callbacks=callbacks)
    y_pred = model.predict(test_ds)


    # https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
    # The class names are represented by a number from 0 to 7 not to a onde hot vector, so its necessary
    # to convert one to another to make possible get the metrics f1 score, recall and precision.
    y_true = [a.numpy() for x,y in test_ds for a in y]
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = to_categorical(y_pred, num_classes = 8, dtype ="int32")
    y_true = to_categorical(y_true, num_classes = 8, dtype ="int32")
    print(classification_report(y_true, y_pred))

# --------------------------- RESULTS --------------------------- 
# Epoch 20/200
# 852/852 [==============================] - 103s 121ms/step - loss: 0.9904 - acc: 0.6708 - val_loss: 1.7178 - val_acc: 0.6596 - lr: 1.0000e-06
#               precision    recall  f1-score   support

#            0       0.97      0.72      0.83        43
#            1       0.34      0.97      0.50        77
#            2       0.95      0.40      0.56        92
#            3       0.83      0.09      0.17        54
#            4       0.94      0.69      0.80        68
#            5       1.00      0.50      0.67        18
#            6       0.94      0.91      0.92        53
#            7       1.00      0.90      0.95        30

#    micro avg       0.64      0.64      0.64       435
#    macro avg       0.87      0.65      0.67       435
# weighted avg       0.83      0.64      0.64       435
#  samples avg       0.64      0.64      0.64       435
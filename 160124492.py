from cgi import test
from tabnanny import verbose
from keras.applications.resnet import ResNet50
from keras.models import Model
from keras.layers import Input, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image_dataset import image_dataset_from_directory
import splitfolders
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

DATASET_REPO = 'raw_dataset'

# F1 score as metrics is based on 
# https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_model():
    # based on 
    # https://github.com/hiagop22/Identify_text_non_text/blob/dev_hiago/text_non_text.py
    # from line 94 to 116
    cnn_model = ResNet50(weights='imagenet', 
                         input_shape=(224,224,3),
                         include_top=False)

    x = Flatten(name='Flatten')(cnn_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(8, activation='softmax')(x)

    model = Model(inputs=cnn_model.input, outputs=x)

    for layer in cnn_model.layers:
        layer.trainable = False

    opt = RMSprop(learning_rate=0.01, )
    # I used 'sparse_categorical_crossentropy' cause the outputs from my
    # dataset arenot one_hot vectors, but integres, according to the bug descripted in 
    # this site https://python.tutorialink.com/i-created-a-cifar10-dataset-learning-model-using-a-cnn-model-why-is-there-an-error/
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['acc',f1_m,precision_m, recall_m])
    model.summary()
    return model

def get_data():
    # splitfolder is based on 
    # https://stackoverflow.com/questions/53074712/how-to-split-folder-of-images-into-test-training-validation-sets-with-stratified
    # cause the raw datataset download isn't already splitted
    splitfolders.ratio(DATASET_REPO, output="processed_dataset", seed=1, ratio=(0.8, 0.1, 0.1))

    train_ds = image_dataset_from_directory(directory='processed_dataset/train',
                                            image_size=(224,224),
                                            labels='inferred',
                                            batch_size=32,
                                            seed=1,
                                            shuffle=True
                                        )

    val_ds = image_dataset_from_directory(directory='processed_dataset/val',
                                            image_size=(224,224),
                                            labels='inferred',
                                            batch_size=32,
                                            seed=1,
                                            shuffle=False
                                        )

    test_ds = image_dataset_from_directory(directory='processed_dataset/test',
                                            image_size=(224,224),
                                            labels='inferred',
                                            batch_size=32,
                                            seed=1,
                                            shuffle=False
                                        )
    return train_ds, val_ds, test_ds

# Callbacks are based on:
# ModelCheckpoint: https://github.com/hiagop22/Identify_text_non_text/blob/dev_hiago/text_non_text.py
# EarlyStopping: https://keras.io/api/callbacks/early_stopping/
# ReduceLROnPlateau: https://keras.io/api/callbacks/reduce_lr_on_plateau/#:~:text=ReduceLROnPlateau%20class&text=Reduce%20learning%20rate%20when%20a,the%20learning%20rate%20is%20reduced.
callbacks = [
    ModelCheckpoint(filepath='best-model.ckpt', 
                    monitor='val_accuracy', 
                    save_best_only=True,
                    mode='max'),
    EarlyStopping(monitor='val_loss',
                  min_delta=0.001,
                  patience=10,
                  mode='min',
                  restore_best_weights=True),
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
    train_ds, val_ds, test_ds = get_data()
    model.fit(train_ds, validation_data=val_ds, epochs=200, verbose=0, callbacks=callbacks)
    loss, accuracy, f1_score, precision, recall = model.evaluate(test_ds, verbose=0)
    print(accuracy, f1_score)
#import umap
import h5py
import numpy  as np
#from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
#from mpl_toolkits.mplot3d import Axes3D
#import pytorch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import keras
from keras import regularizers, optimizers
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization,Input, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras import applications
import cv2
from keras import optimizers
from keras.optimizers import SGD, RMSprop,Adam
from keras.applications import *
from keras.callbacks import *
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint



def create_dataset(my_path):
    l=[]
    for root_, folder_, files_ in os.walk(my_path):
        for f in files_:
            if f.endswith(".png"):
                l.append(os.path.join(root_,f))
    return(l)


def generate_imgs(k, batch_size, root_folder, df):
    X = []
    Y = []
    for i in range(k * batch_size, (k + 1) * batch_size, 1):
        im = plt.imread(os.path.join(root_folder, df[0].iloc[i]))
        #im = (cv2.resize(im, (139, 139)))
        im = (cv2.resize(im, (139, 139))*255).astype("uint8")
        plt.imsave("/cyclope/asanogo/pharma/cell_images/malaria.jpg",im)
        # print(im.shape)
        X.append(im)
        Y.append(df[1].iloc[i])
    Y=to_categorical(Y)

    return np.array(X).reshape(batch_size, X[0].shape[0], X[0].shape[1], X[0].shape[2]), np.array(Y)


def main():
    os.chdir("/cyclope/asanogo/pharma")
    Not_infected = create_dataset("cell_images/Uninfected")
    Infected = create_dataset("cell_images/Parasitized")
    Obs_Not_infected_DF = pd.DataFrame(Not_infected)
    Obs_Infected_DF = pd.DataFrame(Infected)
    Lab_Not_infected_DF = pd.DataFrame(np.repeat(0, pd.DataFrame(Obs_Not_infected_DF).shape[0]))
    Lab_Infected_DF = pd.DataFrame(np.repeat(1, pd.DataFrame(Obs_Infected_DF).shape[0]))

    NI = pd.concat((Obs_Not_infected_DF, Lab_Not_infected_DF), axis=1)
    NI.columns = ["filename", "label"]
    print(NI.head())

    I = pd.concat((Obs_Infected_DF, Lab_Infected_DF), axis=1)
    I.columns = ["filename", "label"]
    print(I.head())

    master_file = pd.concat([NI, I], axis=0)
    master_file = pd.DataFrame(master_file, dtype='str')

    master_file.iloc[0]["filename"]

    ds_ = master_file.sample(frac=1, random_state=1).reset_index()
    fraction = 0.8
    Train = ds_.iloc[0:int(fraction * ds_.shape[0])]
    Test = ds_.iloc[int(fraction * ds_.shape[0]):]

    X_train, y_train, X_test, y_test = Train["filename"], Train["label"], Test["filename"], Test["label"]



    # Model1 = keras.applications.InceptionResNetV2(include_top=False,
    #                                               weights='imagenet',
    #                                               input_tensor=None,
    #                                               input_shape=(139, 139, 3),
    #                                               pooling=None,
    #                                               classes=2)

    Model1 = keras.applications.ResNet50(include_top=False,
                                                  weights='imagenet',
                                                  input_tensor=None,
                                                  input_shape=(139, 139, 3),
                                                  pooling=None,
                                                  classes=2)

    #print(Model1.summary())

    Model1.layers.pop(0)
    RNet_incomplete = Model1
    #RNet_incomplete.summary()

    a = Input(shape=(139, 139, 3))
    b = RNet_incomplete(a)

    b = Flatten()(b)
    b = Dense(2)(b)
    b = Activation('softmax')(b)
    #b = Dense(1)(a)

    Model3 = Model(inputs=a, outputs=b)

    # Model =
    # print(Model3.input_shape)
    #print(Model3.summary())

    n_epochs = 300
    b_size = 256
    save_best_weights = True

    root_f = "/cyclope/asanogo/pharma"
    os.chdir("/cyclope/asanogo/pharma/cell_images/")
    if not os.path.exists("./weights"):
        os.mkdir("./weights")

    if not os.path.exists("./logs"):
        os.mkdir("./logs")

    logger = keras.callbacks.CSVLogger("./logs/logger.csv", separator=',', append=False)
    checkpointer = ModelCheckpoint(filepath='./weights/weights.hdf5',
    verbose=1,
    monitor='val_acc',
    save_best_only=save_best_weights)

    #RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)


    #Model3.compile(optimizer=RMSprop(lr=0.001),
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])

    ModelX = Sequential()
    ModelX.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(139, 139, 3)))
    ModelX.add(Activation('relu'))
    ModelX.add(Conv2D(32, (3, 3)))
    ModelX.add(Activation('relu'))
    ModelX.add(MaxPooling2D(pool_size=(2, 2)))

    ModelX.add(Flatten())
    ModelX.add(Dense(1028))
    ModelX.add(Activation('relu'))
    ModelX.add(Dropout(0.1))
    ModelX.add(Dense(2))
    ModelX.add(Activation('softmax'))




















    Model3.compile(optimizer=SGD(lr=0.001, momentum=0.9),
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

    # ModelX.compile(optimizer=SGD(lr=0.001, momentum=0.9),
    #                loss='sparse_categorical_crossentropy',
    #                metrics=['accuracy'])
    #
    #
    # ModelX.compile(optimizer=RMS(lr=0.001, momentum=0.9),
    #                loss='sparse_categorical_crossentropy',
    #                metrics=['accuracy'])

    ModelX.compile(optimizer=Adam(lr=0.01),
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

    earlyStopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='max')

    os.makedirs('./tensorboard', exist_ok=True)

    tensorboard = keras.callbacks.TensorBoard(log_dir='./tensorboard',
                                              histogram_freq=0,
                                              batch_size=b_size,
                                              write_graph=True,
                                              write_grads=False,
                                              write_images=False,
                                              embeddings_freq=0,
                                              embeddings_layer_names=None,
                                              embeddings_metadata=None,
                                              embeddings_data=None,
                                              update_freq='epoch')

    for epoch in range(n_epochs):
        print("Assans says, training epoch number : {0}".format(epoch))
        try:
            for n in range(int(X_train.shape[0]/b_size)):

                X_train_batch, Y_train_batch = generate_imgs(k=n,
                                                             batch_size=b_size,
                                                             df=[X_train, y_train],
                                                             root_folder=root_f)

                X_test_batch, Y_test_batch = generate_imgs(k=n,
                                                           batch_size=b_size,
                                                           df=[X_test, y_test],
                                                           root_folder=root_f)

                Model3.fit(X_train_batch, Y_train_batch,
                           callbacks=[logger, tensorboard, checkpointer, earlyStopping],
                           validation_data=(X_test_batch, Y_test_batch)
                           )


                Model3.save_weights('./weights/my_model.h5')

        except:
            pass

if __name__ =='__main__':
    main()

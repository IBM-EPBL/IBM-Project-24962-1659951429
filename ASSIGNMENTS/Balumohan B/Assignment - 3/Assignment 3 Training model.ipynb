{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {
    "id": "leb16xLl6UZR",
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "id": "lM_Y4H_a6fGH"
   },
   "outputs": [],
   "source": [
    "train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,\n",
    "                                 zoom_range=0.2,horizontal_flip=True)\n",
    "test_datagen=ImageDataGenerator(rescale=1./255)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "emcSFZNf6oLb",
    "outputId": "fdd803fd-915d-4842-bafe-aa9b78dea4bb"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 4317 images belonging to 5 classes.\n"
     ]
    }
   ],
   "source": [
    "xtrain=train_datagen.flow_from_directory(r'Flowers-Dataset/flowers/train',\n",
    "                                         target_size=(64, 64),\n",
    "                                          batch_size=100,                                           \n",
    "                                          class_mode='categorical'\n",
    "                                         )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "F-maVOCo6yCE",
    "outputId": "bc86a1ed-0be8-44c3-a14e-fb0f18df0f0a"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 27 images belonging to 5 classes.\n"
     ]
    }
   ],
   "source": [
    "xtest=test_datagen.flow_from_directory(r'Flowers-Dataset/flowers/test',\n",
    "                                          target_size=(64, 64),\n",
    "                                          batch_size=100,                                           \n",
    "                                          class_mode='categorical')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {
    "id": "HNTQ9uwS67LB"
   },
   "outputs": [],
   "source": [
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Convolution2D,MaxPool2D,Flatten,Dense"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {
    "id": "RF34Fjk37Bih"
   },
   "outputs": [],
   "source": [
    "model = Sequential()\n",
    "model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(64,64,3))) \n",
    "model.add(MaxPool2D(pool_size=(2,2))) \n",
    "model.add(Flatten()) \n",
    "model.add(Dense(300,activation='relu')) \n",
    "model.add(Dense(150,activation='relu')) \n",
    "model.add(Dense(5,activation='softmax'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {
    "id": "_HuuemWP7Ewg"
   },
   "outputs": [],
   "source": [
    "model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {
    "id": "fhbLviu3IL99"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_5850/371111531.py:1: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.\n",
      "  model.fit_generator(xtrain,\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/30\n",
      "44/44 [==============================] - 12s 273ms/step - loss: 1.4680 - accuracy: 0.4058 - val_loss: 1.0450 - val_accuracy: 0.3704\n",
      "Epoch 2/30\n",
      "44/44 [==============================] - 14s 317ms/step - loss: 1.1067 - accuracy: 0.5527 - val_loss: 1.0041 - val_accuracy: 0.6296\n",
      "Epoch 3/30\n",
      "44/44 [==============================] - 14s 322ms/step - loss: 1.0130 - accuracy: 0.5997 - val_loss: 0.7982 - val_accuracy: 0.6296\n",
      "Epoch 4/30\n",
      "44/44 [==============================] - 17s 382ms/step - loss: 0.9442 - accuracy: 0.6328 - val_loss: 0.7864 - val_accuracy: 0.7037\n",
      "Epoch 5/30\n",
      "44/44 [==============================] - 15s 332ms/step - loss: 0.8792 - accuracy: 0.6623 - val_loss: 0.9963 - val_accuracy: 0.7778\n",
      "Epoch 6/30\n",
      "44/44 [==============================] - 18s 404ms/step - loss: 0.8547 - accuracy: 0.6690 - val_loss: 0.8110 - val_accuracy: 0.7037\n",
      "Epoch 7/30\n",
      "44/44 [==============================] - 16s 350ms/step - loss: 0.7990 - accuracy: 0.7007 - val_loss: 0.6435 - val_accuracy: 0.8148\n",
      "Epoch 8/30\n",
      "44/44 [==============================] - 15s 343ms/step - loss: 0.7546 - accuracy: 0.7142 - val_loss: 0.7294 - val_accuracy: 0.7407\n",
      "Epoch 9/30\n",
      "44/44 [==============================] - 18s 396ms/step - loss: 0.7123 - accuracy: 0.7313 - val_loss: 0.6709 - val_accuracy: 0.7778\n",
      "Epoch 10/30\n",
      "44/44 [==============================] - 18s 407ms/step - loss: 0.6852 - accuracy: 0.7438 - val_loss: 0.6060 - val_accuracy: 0.7778\n",
      "Epoch 11/30\n",
      "44/44 [==============================] - 15s 334ms/step - loss: 0.6674 - accuracy: 0.7417 - val_loss: 0.6927 - val_accuracy: 0.6667\n",
      "Epoch 12/30\n",
      "44/44 [==============================] - 17s 378ms/step - loss: 0.6334 - accuracy: 0.7670 - val_loss: 0.6090 - val_accuracy: 0.7778\n",
      "Epoch 13/30\n",
      "44/44 [==============================] - 15s 338ms/step - loss: 0.6083 - accuracy: 0.7714 - val_loss: 0.4950 - val_accuracy: 0.8519\n",
      "Epoch 14/30\n",
      "44/44 [==============================] - 17s 377ms/step - loss: 0.5675 - accuracy: 0.7855 - val_loss: 0.4593 - val_accuracy: 0.8519\n",
      "Epoch 15/30\n",
      "44/44 [==============================] - 15s 347ms/step - loss: 0.5539 - accuracy: 0.7890 - val_loss: 0.5288 - val_accuracy: 0.8519\n",
      "Epoch 16/30\n",
      "44/44 [==============================] - 15s 334ms/step - loss: 0.5261 - accuracy: 0.8043 - val_loss: 0.4308 - val_accuracy: 0.8889\n",
      "Epoch 17/30\n",
      "44/44 [==============================] - 16s 375ms/step - loss: 0.4869 - accuracy: 0.8205 - val_loss: 0.6826 - val_accuracy: 0.8148\n",
      "Epoch 18/30\n",
      "44/44 [==============================] - 15s 333ms/step - loss: 0.4781 - accuracy: 0.8274 - val_loss: 0.4648 - val_accuracy: 0.8148\n",
      "Epoch 19/30\n",
      "44/44 [==============================] - 15s 344ms/step - loss: 0.4325 - accuracy: 0.8453 - val_loss: 0.2542 - val_accuracy: 0.8889\n",
      "Epoch 20/30\n",
      "44/44 [==============================] - 14s 306ms/step - loss: 0.4402 - accuracy: 0.8404 - val_loss: 0.2697 - val_accuracy: 0.9259\n",
      "Epoch 21/30\n",
      "44/44 [==============================] - 15s 348ms/step - loss: 0.4141 - accuracy: 0.8531 - val_loss: 0.3391 - val_accuracy: 0.7778\n",
      "Epoch 22/30\n",
      "44/44 [==============================] - 16s 375ms/step - loss: 0.4195 - accuracy: 0.8527 - val_loss: 0.3712 - val_accuracy: 0.8148\n",
      "Epoch 23/30\n",
      "44/44 [==============================] - 16s 357ms/step - loss: 0.3524 - accuracy: 0.8816 - val_loss: 0.3969 - val_accuracy: 0.8519\n",
      "Epoch 24/30\n",
      "44/44 [==============================] - 17s 400ms/step - loss: 0.3484 - accuracy: 0.8747 - val_loss: 0.5674 - val_accuracy: 0.8148\n",
      "Epoch 25/30\n",
      "44/44 [==============================] - 14s 321ms/step - loss: 0.3002 - accuracy: 0.8976 - val_loss: 0.4071 - val_accuracy: 0.8519\n",
      "Epoch 26/30\n",
      "44/44 [==============================] - 13s 299ms/step - loss: 0.2927 - accuracy: 0.8967 - val_loss: 0.2987 - val_accuracy: 0.8519\n",
      "Epoch 27/30\n",
      "44/44 [==============================] - 14s 320ms/step - loss: 0.3122 - accuracy: 0.8837 - val_loss: 0.1558 - val_accuracy: 0.9630\n",
      "Epoch 28/30\n",
      "44/44 [==============================] - 15s 340ms/step - loss: 0.2895 - accuracy: 0.8983 - val_loss: 0.2189 - val_accuracy: 0.8889\n",
      "Epoch 29/30\n",
      "44/44 [==============================] - 15s 351ms/step - loss: 0.2702 - accuracy: 0.9092 - val_loss: 0.3741 - val_accuracy: 0.8148\n",
      "Epoch 30/30\n",
      "44/44 [==============================] - 13s 302ms/step - loss: 0.2515 - accuracy: 0.9168 - val_loss: 0.1828 - val_accuracy: 0.8889\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f80dc1ab520>"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit_generator(xtrain,\n",
    "                    steps_per_epoch=len(xtrain),\n",
    "                    epochs=30,\n",
    "                    validation_data=xtest,\n",
    "                    validation_steps=len(xtest))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save('flower.h5')\n",
    "model_json=model.to_json()\n",
    "with open(\"model-bw.json\",\"w\") as json_file:json_file.write(model_json)"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "venv",
   "language": "python",
   "name": "venv"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}

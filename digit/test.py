from numpy import argmax
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

def load_file(file):
    img = load_img(file, target_size=(28, 28), grayscale=True)

    img = img_to_array(img)
    
    img = img.reshape(1, 28, 28, 1)

    img = img.astype('float32')

    img = img/255.0
    return img

def example():  
    img = load_file('th.png')
    model = load_model('model.h5')

    predict_value = model.predict(img)
    digit = argmax(predict_value)

    print(digit)

example()
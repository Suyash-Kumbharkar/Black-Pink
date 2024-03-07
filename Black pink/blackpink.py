import tensorflow as tf
import numpy as np
import os
from os import path, getcwd, chdir
import matplotlib as plt
from tensorflow import keras
from tensorflow.keras.models import load_model

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy')>0.95:
            self.model.stop_training=True
            print("Reached 95% accuracy so cancelling training!")
callbacks=[myCallback()]            

jennie='C:/Users/LENOVO/Desktop/Files/Python/Python/Personalprojects/data/jennie'
lisa='C:/Users/LENOVO/Desktop/Files/Python/Python/Personalprojects/data/lisa'
rose='C://Users/LENOVO/Desktop/Files/Python/Python/Personalprojects/data/rose'
jisoo='C:/Users/LENOVO/Desktop/Files/Python/Python/Personalprojects/data/jisoo'
    
data='C:/Users/LENOVO/Desktop/Files/Python/Python/Personalprojects/data'

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(4,activation='softmax')
    ])


from tensorflow.keras.optimizers import SGD
sgd=SGD(learning_rate=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen =ImageDataGenerator(rotation_range=45, width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(data,target_size=(256,256),color_mode = 'grayscale',class_mode='categorical',shuffle=True,seed=42,batch_size=32)
train_generator=np.array(train_generator)

print(train_generator.class_indices)
a=train_generator.class_indices
keys=a.keys()
values=a.values()
print(keys)
print(values)
history = model.fit(train_generator,epochs=25,callbacks=[myCallback()])
print(history.history['accuracy'][-1])
model.save('C:/Users/LENOVO/Desktop/Files/Python/Python/Personalprojects/blackpink_trained_model/bp_trained_model.h5')
new_model = tf.keras.models.load_model('C:/Users/LENOVO/Desktop/Files/Python/Python/Personalprojects/blackpink_trained_model/bp_trained_model.h5')
print(new_model.summary())
testing_array=[]
testing='C:/Users/LENOVO/Desktop/Files/Python/Python/Personalprojects/testing_blackpink/'
for imj in os.listdir(testing):
    buffer = open(str(testing) + imj,'r')
    testing2=np.array(buffer)
    testing_array.append(testing2) 
    from keras.preprocessing import image
    img = image.load_img('C:/Users/LENOVO/Desktop/Files/Python/Python/Personalprojects/testing_blackpink/'+imj, target_size=(300,300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    main=max(classes)
    index=np.argmax(classes)
    print(main)
    print(list(classes))
    print(index)
    labels=['jennie','jisoo','lisa','rose']
    print(labels[index])  
    print(imj)
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    image = mpimg.imread(testing+imj)
    plt.imshow(image)
    plt.show()
    



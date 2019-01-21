from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model,load_model

model = InceptionV3(weights='imagenet')
model_new = Model(model.input, model.layers[-2].output)
model_new.summary()
model_new.save('models/mod.h5')

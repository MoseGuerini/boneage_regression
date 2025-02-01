#trying to build a CNN

from utils import run_preliminary_test
from cnn import cnn

image_train, real_age_train, gender_train = run_preliminary_test() 

#a sprinkle of preprocessing with matlab
'''Not done yet, too many tears and too little results'''

model = cnn()

'''
model.fit(
    image_train, real_age_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)
'''
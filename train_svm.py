import os
import pickle

import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from car_finder import CarFinder

vehicle_dir = "train_images/vehicles"
non_vehicle_dir = "train_images/non-vehicles"
total_imgs = 0

spatial_size = (64, 64)
n_bins = 128
hist_range = (0, 255)

car_finder = CarFinder(64, hist_bins=128, small_size=20, orientations=12, pix_per_cell=8, cell_per_block=1)

print(car_finder.num_features)
def create_features_from_dir(root_dir):
    all_features = []
    for subdir in (os.listdir(root_dir)):
        for file in os.listdir(os.path.join(root_dir,subdir)):
            img = mpimg.imread(os.path.join(root_dir,subdir,file))
            img = (img.astype(np.float32)/np.max(img)*255).astype(np.uint8)
            features = car_finder.get_features(img)
            all_features.append(features)
    return np.vstack(all_features)

print("Featuring started")
X_cars = create_features_from_dir(vehicle_dir)
y_cars = np.ones(X_cars.shape[0], dtype=np.uint8)
X_non_cars = create_features_from_dir(non_vehicle_dir)
y_non_cars = np.zeros(X_non_cars.shape[0], dtype=np.uint8)
print("Featuring ended")

X = np.vstack((X_cars, X_non_cars))
y = np.concatenate((y_cars, y_non_cars))

scaler = StandardScaler().fit(X)
scaled_X = scaler.transform(X)

del X_cars
del X_non_cars

print(scaled_X.shape)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.10, random_state=40)

cls = LinearSVC(C=1e-4, dual=False, max_iter=5)
cls.fit(X_train, y_train)
result = cls.predict(X_test)

true = y_test == 1
test_len = len(y_test)
positive = result == 1
true_positive = np.sum(true & positive)
true_negative = np.sum(true & np.logical_not(positive))
false_negative = np.sum(np.logical_not(true) & np.logical_not(positive))
false_positive = np.sum(np.logical_not(true) & positive)
print("TP {0}, TN {1}, FN {2}, FP {3}".format(float(true_positive)/test_len, float(true_negative)/test_len,
                                            float(false_negative)/test_len, float(false_positive)/test_len))
score = cls.score(X_test, y_test)
print(score)

data = {
        'scaler': scaler,
        'classifier': cls
        }

with open('classifier.p', 'wb') as f:
    pickle.dump(data, f)
    



from pathlib import Path
import numpy as np
from sklearn import svm
import cv2
import csv

def load_images(path ,csv_file):
    with open(path + '//' + csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        images = []
        labels = []
        next(csv_file)
        for row in csv_reader:
            img = read_image(path + "//" + row[0])
            images.append(img)
            labels.append(row[1])
        return images, labels

def read_image(path):
    img = cv2.imread(path)
    return img
    
def resize_images(imgs, size):
    resized_imgs = []
    for img in imgs:
        resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        resized_imgs.append(resized)
    return resized_imgs
    
def extract_features(imgs):
    hog = cv2.HOGDescriptor()
    hog_features = []
    for img in imgs:
        hog_feature = hog.compute(img)
        hog_features.append(hog_feature)
    return hog_features

def encode_labels(labels):
    labels_map = {'lilyvalley':0, 'tigerlily':1, 'snowdrop':2, 'bluebell':3, 'fritillary':4}
    numeric_labels = []
    for label in labels:
        numeric_labels.append(labels_map[label])
    return numeric_labels

def count_accuracy(result, test_labels):
    counter = 0
    for i in range(len(result)):
        if result[i] == test_labels[i]:
            counter +=1
    return counter/len(result)

if __name__ == '__main__':
    train_img, train_labels = load_images("train", "train_labels.csv")
    test_img, test_labels = load_images("test", "test_labels.csv")

    train_labels_int = encode_labels(train_labels)
    test_labels_int = encode_labels(test_labels)

    train_img = resize_images(train_img,(128,128))
    test_img = resize_images(test_img,(128,128))

    features_train = extract_features(train_img)
    features_test = extract_features(test_img)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-10))
    svm.setC(100)
    svm.setGamma(0.1)
    svm.train(np.array(features_train), cv2.ml.ROW_SAMPLE, np.array(train_labels_int))

    predicted = svm.predict(np.array(features_test, np.float32))

    result = []
    for p in predicted[1]:
        result.append(int(p[0]))

    print("Acurracy: " + str(count_accuracy(result, test_labels_int)))
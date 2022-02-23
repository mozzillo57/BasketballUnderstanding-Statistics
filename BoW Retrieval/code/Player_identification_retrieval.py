import csv
import pickle

import cv2
import numpy as np 
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from scipy import stats

def getFiles(train, path):
    images = []
    count = 0
    for folder in os.listdir(path):
        for file in  os.listdir(os.path.join(path, folder)):
            images.append(os.path.join(path, os.path.join(folder, file)))

    if(train is True):
        np.random.shuffle(images)
    
    return images

def getDescriptors(sift, img):
    kp, des = sift.detectAndCompute(img, None)
    return des

def readImage(img_path):
    img = cv2.imread(img_path, 0)
    return cv2.resize(img,(150,150))

def vstackDescriptors(descriptor_list):
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor)) 

    return descriptors

def clusterDescriptors(descriptors, no_clusters):
    kmeans = KMeans(n_clusters = no_clusters).fit(descriptors)
    return kmeans

def extractFeatures(kmeans, descriptor_list, image_count, no_clusters):
    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1

    return im_features

def normalizeFeatures(scale, features):
    return scale.transform(features)

def plotHistogram(im_features, no_clusters):
    x_scalar = np.arange(no_clusters)
    y_scalar = np.array([abs(np.sum(im_features[:,h], dtype=np.int32)) for h in range(no_clusters)])

    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Complete Vocabulary Generated")
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.show()


def trainModel(path, no_clusters):
    images = getFiles(True, path)
    images_path = []
    print("Train images path detected.")
    sift = cv2.SIFT_create()
    descriptor_list = []
    train_labels = np.array([])
    label_count = 3
    image_count = len(images)

    for img_path in images:
        if("curry" in img_path):
            class_index = 0
        elif("lebron" in img_path):
            class_index = 1
        elif("duncan" in img_path):
            class_index = 2
        elif ("giannis" in img_path):
            class_index = 3

        train_labels = np.append(train_labels, class_index)
        img = readImage(img_path)
        des = getDescriptors(sift, img)
        images_path.append(img_path)
        descriptor_list.append(des)

    descriptors = vstackDescriptors(descriptor_list)
    print("Descriptors vstacked.")

    kmeans = clusterDescriptors(descriptors, no_clusters)
    print("Descriptors clustered.")

    im_features = extractFeatures(kmeans, descriptor_list, image_count, no_clusters)
    print("Images features extracted.")

    scale = StandardScaler().fit(im_features)        
    im_features = scale.transform(im_features)
    print("Train images normalized.")

    plotHistogram(im_features, no_clusters)
    print("Features histogram plotted.")

    print("Training completed.")
    print(f'train labels {train_labels}')

    return kmeans, scale, im_features, images_path

def testModel(path, kmeans, scale, im_features, no_clusters, images_path):
    test_images = getFiles(False, path)
    print("Test images path detected.")

    count = 0
    descriptor_list = []

    # name_dict =	{
    #     "0": "curry",
    #     "1": "lebron",
    #     "2": "duncan",
    #     "3": "giannis"
    # }

    sift = cv2.SIFT_create()

    query_path = ''
    for img_path in test_images:
        img = readImage(img_path)
        query_path = img_path
        des = getDescriptors(sift, img)

        if(des is not None):
            count += 1
            descriptor_list.append(des)

    descriptors = vstackDescriptors(descriptor_list)

    test_features = extractFeatures(kmeans, descriptor_list, count, no_clusters)
    print(f'feature prima di scaling {test_features}')
    test_features = scale.transform(test_features)
    print(test_features)
    print(test_features.shape)

    print(f'im_features {im_features}')
    print(im_features.shape)

    cont = 0
    result_dist = 100
    result_index = -1
    for test in test_features:
        print(f'feat {test}')
        print(f'feat shape {test.shape}')
        for feat in im_features:
            dist_candidate = stats.wasserstein_distance(feat, test)
            print(f'dist candidate: {dist_candidate}')
            if dist_candidate < result_dist:
                result_dist = dist_candidate
                result_index = cont
            cont += 1
        print(result_index)
        print(images_path[result_index])

        result = cv2.imread(images_path[result_index])
        query = readImage(query_path)
        cv2.imshow("query", query)
        print(query_path)
        cv2.imshow("result", result )
        cv2.waitKey(0)

        # closing all open windows
        cv2.destroyAllWindows()





def execute_saving_datas(train_path, test_path, no_clusters):
    kmeans, scale, im_features, images_path = trainModel(train_path, no_clusters)
    print(f'kmeans: {kmeans}')
    print(f'scale: {scale}')
    print(f'im_features: {im_features}')
    print(f'images_path: {images_path}')

    # save the kmeans to disk
    filename = '../local_data/kmeans.sav'
    pickle.dump(kmeans, open(filename, 'wb'))

    # save the standard scaler to disk
    filename = '../local_data/scale.sav'
    pickle.dump(scale, open(filename, 'wb'))

    with open('../local_data/im_features.csv', 'x', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(im_features)
    with open('../local_data/images_path.csv', 'x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(images_path)

    testModel(test_path, kmeans, scale, im_features, no_clusters, images_path)

def execute_from_local(test_path, no_clusters):

    filename = '../local_data/kmeans.sav'
    kmeans = pickle.load(open(filename, 'rb'))
    print(f'kmeans: {kmeans}')

    filename = '../local_data/scale.sav'

    scale = pickle.load(open(filename, 'rb'))
    print(f'scale: {scale}')

    im_features = []
    with open('../local_data/im_features.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        for row in csv_reader:
            riga = []
            for el in row:
                riga.append(np.float32(el))
            im_features.append(riga)
    im_features = np.array(im_features)
    print(f'im_features: {im_features}')

    images_path = []
    with open('../local_data/images_path.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        for row in csv_reader:
            images_path = row
        print(f'images_path: {images_path}')

    testModel(test_path, kmeans, scale, im_features, no_clusters, images_path)

def execute_live(train_path, test_path, no_clusters):
    kmeans, scale, im_features, images_path = trainModel(train_path, no_clusters)

    testModel(test_path, kmeans, scale, im_features, no_clusters, images_path)

if __name__ == '__main__':
    #execute_live('../dataset/train', '../dataset/test_giannis', 100)

    execute_saving_datas('../dataset/train', '../dataset/test_giannis', 50)

    execute_from_local('../dataset/test_curry', 50)

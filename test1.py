import os
import keras
from keras.api.preprocessing import image
from keras.api.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.api.models import Model
import numpy as np
import time
from scipy.spatial import distance
import matplotlib.pyplot as plt


path ="/home/ivan/pythonCode/readText/pictures" 
filelist = [] 
for root, dirs, files in os.walk(path): 
    for file in files: 
        filelist.append(os.path.join(root,file))


model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
model.summary()


def load_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
feat_extractor.summary()


tic = time.perf_counter()
features = []
for i, image_path in enumerate(filelist[:9]):
    if i % 500 == 0:
        toc = time.perf_counter()
        elap = toc-tic
        print("analyzing image %d / %d. Time: %4.4f seconds." % (i, len(features),elap))
        tic = time.perf_counter()
    img, x = load_image(image_path)
    feat = feat_extractor.predict(x)[0]
    features.append(feat)
print('finished extracting features for %d images' % len(features))

from sklearn.decomposition import PCA
features = np.array(features)
pca = PCA(n_components=9)
pca.fit(features)

pca_features = pca.transform(features)


similar_idx = [ distance.cosine(pca_features[3], feat) for feat in pca_features ]

idx_closest = sorted(range(len(similar_idx)), key=lambda k: similar_idx[k])[0:6]


thumbs = []
for idx in idx_closest:
    img = image.load_img(filelist[idx])
    img = img.resize((int(img.width * 100 / img.height), 100))
    thumbs.append(img)

# concatenate the images into a single image
concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)

# show the image
plt.figure(figsize = (16,12))
plt.imshow(concat_image)
plt.show()

print("OK")
# In[1]:
import numpy as np
import pandas as pd

# In[2]:
from keras.preprocessing.image import load_img, img_to_array

# In[3]:

img_width, img_height = 256, 256

# In[4]:
def preprocess_image(path):
    img = load_img(path, target_size = (img_height, img_width))
    a = img_to_array(img)
    a = np.expand_dims(a, axis = 0)
    a /= 255.
    return a

# In[5]:
test_images_dir = 'test/'

# In[6]:
test_df = pd.read_csv('test.csv')

test_dfToList = test_df['Image_id'].tolist()
test_ids = [str(item) for item in test_dfToList]

# In[7]:

test_images = [test_images_dir+item for item in test_ids]
test_preprocessed_images = np.vstack([preprocess_image(fn) for fn in test_images])

# In[8]:
np.save('../test_preproc_CNN.npy', test_preprocessed_images)



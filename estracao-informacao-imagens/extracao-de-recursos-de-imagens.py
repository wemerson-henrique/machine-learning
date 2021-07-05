import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

# Image manipulation.
import PIL.Image
from IPython.display import display

# importing required modules
from zipfile import ZipFile

# specifying the zip file name
images_zip = '../input/leaf-classification/images.zip'

# opening the zip file in READ mode
with ZipFile(images_zip, 'r') as zip:
    # printing all the contents of the zip file
    #     zip.printdir()

    # extracting all the files
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')
image_dir = './images' # From output folder.
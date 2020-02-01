import glob
import numpy as np
from PIL import Image

file_list = glob.glob('/media/chengjia/cheng_hd/data/frozen/train/lymphoma/0000317*.tif')

for f in file_list:
    print(f)
    im = Image.open(f)

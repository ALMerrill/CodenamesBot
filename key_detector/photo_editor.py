#!/usr/bin/env python3
import os
from PIL import Image
from resizeimage import resizeimage
import numpy as np
import cv2
base = './data/imgs'
output='./data/temp'
upper='./data'
pa='./'
# for path in os.listdir(base):

#     with Image.open(os.path.join(base, path)) as image:
#         # cover = resizeimage.resize_cover(image, [128, 128])
#         image=image.rotate(-90)
#         image.save(os.path.join(output,path), "png")
# for path in os.listdir(base):
#     with Image.open(os.path.join(base,path)) as orig:
#         with Image.open(os.path.join(output,path)) as new:
#             temp=np.concat(orig,new)
#             Image.show(temp)
path='./00212a.png'
with Image.open(os.path.join(path)) as image:
    # image=image.rotate(-90)
    # image.save(os.path.join(base,path), "png")
    # rgb_image = image.convert('RGB')

    print(np.array(image).shape)
    # image.save(os.path.join(upper,path))
    
    # image.load() # required for png.split()

    # background = Image.new("RGB", image.size, (255, 255, 255))
    # background.paste(image, mask=image.split()[3]) # 3 is the alpha channel

    # background.save('00212a.png', 'png')
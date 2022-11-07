#Compute the mean of all images in train split of VoxCeleb2 and test split of VoxCeleb1 together

from PIL import Image
import numpy as np
import os

list_img = []

# for root, dirs, files in os.walk('/raid/madina_abdrakhmanova/datasets/VoxCeleb1/test'):
#     for file in files:    
#         if file.endswith('.jpg'):    
#                 list_img.append(os.path.join(root, file))

for root, dirs, files in os.walk('/raid/madina_abdrakhmanova/datasets/sf_pv/data'):
    for file in files:    
        if file.endswith('.png'):    
            list_img.append(os.path.join(root, file))
            
for root, dirs, files in os.walk('/raid/madina_abdrakhmanova/datasets/VoxCeleb2/dev'):
    for file in files:    
        if file.endswith('.jpg'):    
            list_img.append(os.path.join(root, file))
r = []
g = []
b = []

size = len(list_img)

count = 0
for image in list_img:
    arr = np.array(Image.open(image)) #creating arrays for all the images
#     breakpoint()
    if count % 10000 == 0:
        print(f'{count} / {size}')
    r += np.asarray(arr[:,:,0]).tolist()
    g += np.asarray(arr[:,:,1]).tolist()
    b += np.asarray(arr[:,:,2]).tolist()
    count += 1

print('Mean R: ', sum(r)/len(r))
print('Mean G: ', sum(g)/len(g))
print('Mean B: ', sum(b)/len(b))
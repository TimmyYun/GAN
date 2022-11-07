import numpy as np
from PIL import Image
import os
import time 

filenames = []

# for root, dirs, files in os.walk('/raid/madina_abdrakhmanova/datasets/VoxCeleb1/test'):
#     for file in files:    
#         if file.endswith('.jpg'):    
#                 filenames.append(os.path.join(root, file))
            
for root, dirs, files in os.walk('/raid/madina_abdrakhmanova/datasets/sf_pv/data_v3'):
    for file in files:    
        if file.endswith('.jpg'):    
            filenames.append(os.path.join(root, file))
            
# for root, dirs, files in os.walk('/raid/madina_abdrakhmanova/datasets/VoxCeleb2/dev'):
#     for file in files:    
#         if file.endswith('.jpg'):    
#             filenames.append(os.path.join(root, file))

r = 0
g = 0
b = 0

size = len(filenames)
img_size = (124,124)
count = 0
countR = 0
countG = 0
countB = 0
lastTime = time.time()
beginningTime = time.time()

for path in filenames:
    count += 1
    arr = np.array(Image.open(path).resize(img_size))
    
    if count % 10000 == 0:        
        print(f'{count} / {size}')  
        print(f'Total time: {time.time() - beginningTime} sec. Since last log: {time.time() - lastTime} sec.')
        lastTime = time.time()
        print('Mean R: ', r / countR)
        print('Mean G: ', g / countG)
        print('Mean B: ', b / countB)
    for item in np.asarray(arr[:,:,0]).tolist():
        for value in item:
            countR += 1
            r += value
        
    for item in np.asarray(arr[:,:,1]).tolist():
        for value in item:
            countG += 1
            g += value
        
    for item in np.asarray(arr[:,:,2]).tolist():
        for value in item:
            countB += 1
            b += value
    
print(f'{count} / {size}')
print('Final mean R: ', r / countR)
print('Final mean G: ', g / countG)
print('Final mean B: ', b / countB)


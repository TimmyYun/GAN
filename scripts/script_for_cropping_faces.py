from retinaface import RetinaFace
import cv2
import pandas as pd
meta=pd.read_csv('/home/admin2/datasets/voxceleb1/metadata/vox1_meta.csv')
idnames=[]
for i in meta.iloc[:,0]:
    a=i.find("id1")
    idnames.append(i[a:7])

devset=[]
for i in meta.iloc[:,0]:
    b=i.find("dev")
    if b==-1:
        devset.append('test')
    else:
        devset.append(i[b:])
namesset=[]
for i in meta.iloc[:,0]:
    iterator=0
    c=i[8:].find('\t')
    b=i[8:].split('\t')[0]
    namesset.append(b)
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from PIL import Image as im
import PIL
from numpy import asarray
from matplotlib import pyplot as plt
actors=list()
video=list()
list_videos=list()
a=f"/home/admin2/datasets/voxceleb1/unzippedIntervalFaces/data"
for names_actors in os.listdir(a):
    actors=list(os.listdir(a))
    path_to_actorvids = os.path.abspath(a+"/{}/1.6".format(names_actors))
    if (names_actors in namesset[269:309])==True:
        id_index=namesset.index(names_actors)
        for videos in os.listdir(path_to_actorvids):
            list_videos=list(os.listdir(path_to_actorvids))
            path_to_clips=os.path.abspath(path_to_actorvids+"/{}".format(videos))
            for clip in os.listdir(path_to_clips):
                clip_list=list(os.listdir(path_to_clips+"/{}".format(clip)))

                path_to_save1=f"/home/admin2/datasets/voxceleb1/face_data/retinaface/test"+"/{}".format(idnames[id_index])+"/{}".format(videos)+"/rgb/{}".format(clip)
                if not os.path.exists(path_to_save1):
                    os.makedirs(path_to_save1)

                for images in clip_list:
                    path_to_image=str(path_to_clips+"/{}".format(clip)+"/{}".format(images))
                    faces=RetinaFace.extract_faces(img_path=(f""+path_to_image),align=True)
                    try:
                        face_as_array_reshaped = faces[0]
                        img=im.fromarray(face_as_array_reshaped, 'RGB')
                        img.save(f"/home/admin2/datasets/voxceleb1/face_data/retinaface/test"+"/{}".format(idnames[id_index])+"/{}".format(videos)+"/rgb/{}".format(clip)+"/{}".format(images))
                        img.show()   
                    except IndexError:
                        face_as_array_reshaped = 'null'
    else:
        id_index2=namesset.index(names_actors)
        for videos in os.listdir(path_to_actorvids):
            list_videos=list(os.listdir(path_to_actorvids))
            path_to_clips=os.path.abspath(path_to_actorvids+"/{}".format(videos))
            for clip in os.listdir(path_to_clips):
                clip_list=list(os.listdir(path_to_clips+"/{}".format(clip)))

                path_to_save=f"/home/admin2/datasets/voxceleb1/face_data/retinaface/dev"+"/{}".format(idnames[id_index2])+"/{}".format(videos)+"/rgb/{}".format(clip)
                if not os.path.exists(path_to_save):
                    os.makedirs(path_to_save)

                    for images in clip_list:
                        path_to_image=str(path_to_clips+"/{}".format(clip)+"/{}".format(images))
                        faces=RetinaFace.extract_faces(img_path=(f""+path_to_image),align=True)
                        try:
                            face_as_array_reshaped = faces[0]
                            img=im.fromarray(face_as_array_reshaped, 'RGB')
                            img.save(f"/home/admin2/datasets/voxceleb1/face_data/retinaface/dev"+"/{}".format(idnames[id_index2])+"/{}".format(videos)+"/rgb/{}".format(clip)+"/{}".format(images))
                            img.show()   
                        except IndexError:
                            face_as_array_reshaped = 'null'


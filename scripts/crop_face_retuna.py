import pandas as pd
from retinaface import RetinaFace
from PIL import Image as im
import  os
import glob
meta_df=pd.read_csv('/home/admin2/datasets/voxceleb1/metadata/vox1_meta.csv', sep='\t')

data_path="/home/admin2/datasets/voxceleb1/unzippedIntervalFaces/data"
dest_path="/home/admin2/datasets/voxceleb1/face_data/retinaface_madina/"
meta_df=meta_df[meta_df['Set']=='test']
print(meta_df.shape)
for i in range(len(meta_df)): 
    sub_path=os.path.join(data_path,meta_df.iloc[i,1],'1.6')
    file_paths=glob.glob(sub_path+'/*/*/*.jpg')
    for file_path in file_paths:
        
        file_info=file_path.split(os.sep)
        dest_dir_path=os.path.join(dest_path, "test", meta_df.iloc[i, 0], file_info[9],"rgb",file_info[10])
        
        if not os.path.exists(dest_dir_path):
            os.makedirs(dest_dir_path)
            print("directory created: "+dest_dir_path)
        faces=RetinaFace.extract_faces(img_path=file_path,align=True)
            
        if len(faces)!=0:
            face_arr=faces[0]
            face_img=im.fromarray(face_arr, 'RGB')

            new_file_path = os.path.join(dest_dir_path, file_info[-1])
            face_img.save(new_file_path)
        else:
            print("no faces detected for image {}".format(file_path))

    print("processed {}%".format((i+1)/len(meta_df)*100))

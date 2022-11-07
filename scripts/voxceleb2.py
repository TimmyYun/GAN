import argparse
import os
import random

def getClips(opt):
    
    random.seed(10)
    
    subjects = []
    clips = []
    
    for subject in os.listdir(opt.dir):
        if 'id' in subject:
            subjects.append(subject)
    if opt.percentage == 100:
        numberToProcess = 5994
    else:
        numberToProcess =  round(len(subjects) * opt.percentage * 0.01)
    
    if numberToProcess > len(subjects):
        numberToProcess = len(subjects)
            
    subjectsToProcess = set()
    
    while(len(subjectsToProcess) < numberToProcess):
        subjectsToProcess.add(subjects[random.randint(0, len(subjects)) - 1])
    
    print(subjectsToProcess)
    print(len(subjectsToProcess))
        
    with open(f'/raid/timur_unaspekov/scripts/subjects_{opt.percentage}_{opt.clip}.txt', 'w') as output:
        for subject in sorted(subjectsToProcess):
            output.write(subject + '\n')
    
    
    for subject in sorted(subjectsToProcess):
        for video in os.listdir(os.path.join(opt.dir,subject)):
            path_to_clips = os.path.join(opt.dir, subject, video) + '/rgb/'
            local_clips = []
            
            for clip in os.listdir(path_to_clips):
                local_clips.append(clip)
            local_clips = sorted(local_clips)
            
            clips_to_process = opt.clip
            
            if clips_to_process > len(local_clips):
                clips_to_process = len(local_clips)
            
            for clip_ind in range(0, clips_to_process):
                
                jpgCounter = 0
                for root, dirs, files in os.walk(os.path.join(path_to_clips, sorted(local_clips)[clip_ind])):
                    for file in files:    
                        if file.endswith('.jpg'):
                            jpgCounter += 1
                            
                if jpgCounter == 0:                    
                    print(os.path.join(path_to_clips, sorted(local_clips)[clip_ind]))
                else:
                    clips.append(os.path.join(path_to_clips, sorted(local_clips)[clip_ind]))
                    
                    
                    
    with open(opt.out, 'w') as output:
        for row in clips:
            sub = str(row).split('/')[-4]
            clip = str(row).split('/')[-1].zfill(5) + '.wav'
            row = '/'.join(str(row).split('/')[-4:-1]).replace('rgb' ,'wav')
            output.write(sub + ' VoxCeleb2/dev/' + str(row) + '/' + clip + '\n')
#             output.write(sub + ' ' + str(row) + '/' + clip + '\n')
    
def main(opt):
    getClips(opt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='/raid/madina_abdrakhmanova/datasets/VoxCeleb2/dev', help='directory to dataset')
    parser.add_argument('--percentage', type=int, default='1', help='percentage of dataset')
    parser.add_argument('--clip', type=int, default='1', help='clips to get')  
    parser.add_argument('--out', type=str, default='./out.txt', help='output file')

    opt = parser.parse_args()
    main(opt)
    

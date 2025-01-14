#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import numpy
import random
import pdb
import os
import threading
import time
import math
import glob
import soundfile
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from PIL import Image
from torchvision.transforms import GaussianBlur
from kornia.filters import motion_blur
import sys

def round_down(num, divisor):
    return num - (num % divisor)


def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)


def rgb_augment(x, t, ax=None, k=7, s=(5, 7)):
    title_ = ""

    # vertical blur
    if t == 0:
        direction_ = numpy.random.uniform(-1, 1)
        title_ = f"Vertical, direction={direction_:0.3f}, k={k}"

        ret_ = motion_blur(x, kernel_size=k, angle=90, direction=direction_)
    # horizontal blur
    elif t == 1:
        direction_ = numpy.random.uniform(-1, 1)
        title_ = f"Horizontal, direction={direction_:0.3f}, k={k}"

        ret_ = motion_blur(x, kernel_size=k, angle=180, direction=direction_)

    elif t == 2:
        title_ = f"Gaussian, k={k}, s={s}"
        ret_ = GaussianBlur(kernel_size=k, sigma=s)(x)

    if ax is not None:
        ax.imshow(ret_[0].permute(1, 2, 0) + 0.5)
        ax.set_title(title_)
        ax.set_xticks([])
        ax.set_yticks([])

    return ret_

def rgb_augment_eval(x, t, direction_, ax=None, k=7, s=(5, 7)):
    title_ = ""

    # vertical blur
    if t == 0:
        title_ = f"Vertical, direction={direction_:0.3f}, k={k}"

        ret_ = motion_blur(x, kernel_size=k, angle=90, direction=direction_)
    # horizontal blur
    elif t == 1:
        title_ = f"Horizontal, direction={direction_:0.3f}, k={k}"

        ret_ = motion_blur(x, kernel_size=k, angle=180, direction=direction_)

    elif t == 2:
        title_ = f"Gaussian, k={k}, s={s}"
        ret_ = GaussianBlur(kernel_size=k, sigma=s)(x)

    if ax is not None:
        ax.imshow(ret_[0].permute(1, 2, 0) + 0.5)
        ax.set_title(title_)
        ax.set_xticks([])
        ax.set_yticks([])

    return ret_

def thr_augment(x, t, ax=None, k=7, s=(5, 7)):
    title_ = f"Gaussian, k={k}"
    ret_ = GaussianBlur(kernel_size=k, sigma=s)(x)

    if ax is not None:
        ax.imshow(ret_[0].permute(1, 2, 0) + 0.5)
        ax.set_title(title_)
        ax.set_xticks([])
        ax.set_yticks([])

    return ret_


def snr(signal, noise):
    signal = numpy.sqrt(numpy.mean(signal ** 2))
    noise = numpy.sqrt(numpy.mean(noise ** 2))

    return 10 * numpy.log(signal / noise)


def wav_augment(x, t, musan_path, snr, evalmode):
    t = ['music', 'noise', 'speech'][t]
    path = os.path.join(musan_path, t, "*", "*", "*.wav")
    all_noises = glob.glob(path)
    random_choice = numpy.random.randint(len(all_noises))
    t_audio = all_noises[random_choice]

    clean_db = 10 * numpy.log10(numpy.mean(x ** 2) + 1e-4)
    noiseaudio = loadWAV(t_audio, 200, evalmode=evalmode)
    noise_snr = snr
    noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2) + 1e-4)
    noise = numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio

    return noise + x


def wav_augment_eval(x, t_audio, snr, evalmode):

    clean_db = 10 * numpy.log10(numpy.mean(x ** 2) + 1e-4)
    noiseaudio = loadWAV(t_audio, 200, evalmode=evalmode)
    noise_snr = snr
    noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2) + 1e-4)
    noise = numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio

    return noise + x

def noise_evaluation_set(modality, noise_data, wav, rgb, thr, musan_path, evalmode, p_noise=0.3, snr=0, k=3):
    #
    # Inputs should be tensors
    #
    data = [wav, rgb, thr]
    n, t, mode, t_audio, direction = float(noise_data[0]), int(noise_data[1]), int(noise_data[2]), noise_data[3], float(noise_data[4])

    if n < p_noise:
        if t == k + 1:
            # the random tensor is sampled from normal distribution ~N(0,1)
            data[mode] = torch.randn(size=data[mode].size())
        else:
            if mode == 0:
                data[mode] = wav_augment_eval(data[mode], t_audio, snr, evalmode)
            elif mode == 1:
                data[mode] = rgb_augment_eval(data[mode], t-1, direction)
            else:
                data[mode] = thr_augment(data[mode], t-1)
    return (*data,)

def noise_train_set(modality, wav, rgb, thr, musan_path, evalmode, p_noise=0.3, snr=0, k=3):
    #
    # Inputs should be tensors
    #
    data = [wav, rgb, thr]
    augment_functions = [wav_augment, rgb_augment, thr_augment]
    n = numpy.random.rand()

    if n < p_noise:
        #trimodal case: randomly select modality type; unimodal case: fix the modality
        if "wav" in modality and "rgb" in modality and "thr" in modality:
            mode = numpy.random.randint(3)
        elif "wav" in modality and "rgb" in modality:
            mode = numpy.random.randint(2)
        elif modality == "wav":
            mode = 0
        elif modality == "rgb":
            mode = 1
        elif modality == "thr":
            mode = 2
        else:
            print("Incorrect data stream! Terminating")
            exit()

        #randomly select the noise type from [1, k+1] via uniform distribution
        t = numpy.random.randint(k) + 1
        if t == k + 1:
            # the random tensor is sampled from normal distribution ~N(0,1)
            data[mode] = torch.randn(size=data[mode].size())
        else:
            if mode == 0:
                data[mode] = augment_functions[mode](data[mode], t-1, musan_path, snr,  evalmode)
            else:
                data[mode] = augment_functions[mode](data[mode], t-1)

    return (*data,)

def get_img_list(p, max_images):

    l = glob.glob(p)
    tmp = sorted(l, key=lambda f:int(f.split(os.sep)[-1].split('.')[0]))
    if len(tmp) < max_images:
        max_images = len(tmp)
    if max_images == 1:
        return [tmp[0]]
    else:
        tmp = [tmp[i] for i in numpy.linspace(0, len(tmp) - 1, endpoint=True, num=max_images, dtype=int).tolist()]
        return tmp

def loadWAV(filename, max_frames, evalmode=True, num_eval=10):
    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
#     audio, sample_rate = soundfile.read(filename.replace('VoxCeleb2/dev', 'VoxCeleb1/test'))
    audio, sample_rate = soundfile.read(filename)
    
    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = numpy.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]


    if evalmode:
        startframe = numpy.linspace(0, audiosize - max_audio, num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random() * (audiosize - max_audio))])

    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])

    feat = numpy.stack(feats, axis=0).astype(float)

    return feat;


def loadIMG(filenames, img_size, mean_train=[0, 0, 0]):
    # read images
    images = [Image.open(image) for image in filenames]

    # reduce image dimension
    images_tmp = [image.resize(img_size) for image in images]

    # convert array list into numpy
    images_array = numpy.stack(images_tmp, axis=0).astype(numpy.float32)

    # mean-normalize the images
    for i in range(3):
        images_array[:, :, :, i] = (images_array[:, :, :, i] - mean_train[i]) / 255
    return images_array



def load_train_lists(path):

    with open(os.path.join(path, "wav_list.txt")) as f:
        wav_list = f.read().split()
    
    with open(os.path.join(path, "rgb_list.txt")) as f:
        rgb_list = [line.split() for line in f.readlines()]
    
    with open(os.path.join(path, "thr_list.txt")) as f:
        thr_list = [line.split() for line in f.readlines()]
    
    with open(os.path.join(path, "label_list.txt")) as f:
        label_list = list(map(int, f.read().split()))
    
    return wav_list, rgb_list, thr_list, label_list

class train_dataset_loader(Dataset):
    def __init__(self, train_list, musan_path, max_frames, train_path, train_lists_save_path,
                 **kwargs):
        # for SF only
#         self.mean_rgb = [92.19873, 70.86596, 62.344334]
        
        # for VC2 only
        #self.mean_rgb = [143.0115, 98.5479, 87.7612]
        
        # for SF and VC2 only
        self.mean_rgb = [117.1321, 84.0650, 74.4072]        
        
        self.mean_thr = [241.39543, 189.38396, 76.27353]
        self.noisy_train = kwargs["noisy_train"]
        self.train_list = train_list
        self.max_frames = max_frames
        self.musan_path = musan_path
        self.modality = kwargs["modality"].lower()
        self.num_images = kwargs["num_images"]
        self.img_size = (kwargs["image_width"], kwargs["image_height"])
        self.p_noise = kwargs["p_noise"]
        self.snr = kwargs["snr"]

        print("Initializing the train_data_loader")

        # Read training files
        with open(train_list) as dataset_file:
            lines = dataset_file.readlines();

        # Make a dictionary of ID names and ID indices
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}

        # Parse the training list into file names and ID indices
        self.data_list = []  # audio
        self.data_list_rgb = []  # visual
        self.data_list_thr = []  # thermal
        self.data_label = []

        # Load saved lists if available
        train_lists_save_path = os.path.join(train_lists_save_path, "num_images_" + str(self.num_images))
        if os.path.exists(train_lists_save_path):
            self.data_list, self.data_list_rgb, self.data_list_thr, self.data_label = load_train_lists(
                train_lists_save_path)
            return

        # Create lists for rgb, thr, wav
        for lidx, line in enumerate(lines):
            data = line.strip().split();

            # label
            speaker_label = dictkeys[data[0]];
            self.data_label.append(speaker_label)

            # wav
            filename = os.path.join(train_path, data[1])
            self.data_list.append(filename)
            
            # rgb
            clip_id = str(int(data[1].split('/')[-1].split('.')[0]))
            path_rgb = os.path.join(train_path, os.sep.join(data[1].split(os.sep)[:-2]), "rgb", clip_id, "*.jpg")
            img_list = get_img_list(path_rgb, self.num_images)
            self.data_list_rgb.append(img_list)
            
#             # thr
            path_thr = os.path.join(train_path, os.sep.join(data[1].split(os.sep)[:-2]), "thr", clip_id, "*.jpg")
            img_list = get_img_list(path_thr, self.num_images)
            self.data_list_thr.append(img_list)

            assert len(self.data_list_rgb[-1]) == len(self.data_list_thr[-1]), \
                "number of frames in rgb and thr are not equal in: " + filename
            
        os.makedirs(train_lists_save_path)
        with open(os.path.join(train_lists_save_path, 'wav_list.txt'), 'w') as f:
            f.write(" ".join(self.data_list))

        with open(os.path.join(train_lists_save_path, 'rgb_list.txt'), 'w') as f:
            for line in self.data_list_rgb:
                f.write(" ".join(line) + "\n")

        with open(os.path.join(train_lists_save_path, 'thr_list.txt'), 'w') as f:
            for line in self.data_list_thr:
                f.write(" ".join(line) + "\n")

        with open(os.path.join(train_lists_save_path, 'label_list.txt'), 'w') as f:
            f.write(" ".join(map(str, self.data_label)))

    def __getitem__(self, indices):

        feat = []
        feat_rgb = []
        feat_thr = []

        for index in indices:

            if "wav" in self.modality:
                audio = loadWAV(self.data_list[index], self.max_frames, evalmode=False)
                feat.append(audio);

            if "rgb" in self.modality:
                image = loadIMG(self.data_list_rgb[index], self.img_size,
                                mean_train=self.mean_rgb)
                feat_rgb.append(image)

            if "thr" in self.modality:
                image = loadIMG(self.data_list_thr[index], self.img_size,
                                mean_train=self.mean_thr)
                feat_thr.append(image)

        if "wav" in self.modality and "rgb" in self.modality and "thr" in self.modality:

            if self.noisy_train:
                noisy_feat = []
                noisy_feat_rgb = []
                noisy_feat_thr = []
                for i in range(len(feat)):
                    data = noise_train_set(self.modality, feat[i], torch.FloatTensor(feat_rgb[i]).permute(0, 3, 1, 2), torch.FloatTensor(feat_thr[i]).permute(0, 3, 1, 2),
                                                self.musan_path, False, self.p_noise, self.snr)
                    noisy_feat.append(data[0])
                    noisy_feat_rgb.append(data[1])
                    noisy_feat_thr.append(data[2])
                noisy_feat = numpy.concatenate(noisy_feat, axis=0)
                return (torch.FloatTensor(noisy_feat), torch.cat(noisy_feat_rgb, axis=0), torch.cat(noisy_feat_thr, axis=0)), self.data_label[index]
            else:
                feat = numpy.concatenate(feat, axis=0)
                feat_rgb = numpy.concatenate(feat_rgb, axis=0)
                feat_rgb = torch.FloatTensor(feat_rgb).permute(0, 3, 1, 2)
                feat_thr = numpy.concatenate(feat_thr, axis=0)
                feat_thr = torch.FloatTensor(feat_thr).permute(0, 3, 1, 2)
                return (torch.FloatTensor(feat), feat_rgb, feat_thr), self.data_label[index]

        elif "wav" in self.modality and "rgb" in self.modality:
            if self.noisy_train:
                noisy_feat = []
                noisy_feat_rgb = []
                for i in range(len(feat)):
                    data = noise_train_set(self.modality, feat[i], torch.FloatTensor(feat_rgb[i]).permute(0, 3, 1, 2), None, self.musan_path, False, self.p_noise, self.snr)
                    noisy_feat.append(data[0])
                    noisy_feat_rgb.append(data[1])
                noisy_feat = numpy.concatenate(noisy_feat, axis=0)
                return (torch.FloatTensor(noisy_feat), torch.cat(noisy_feat_rgb, axis=0)), self.data_label[index]
            else:
                feat = numpy.concatenate(feat, axis=0)
                feat_rgb = numpy.concatenate(feat_rgb, axis=0)
                feat_rgb = torch.FloatTensor(feat_rgb).permute(0, 3, 1, 2)
                return (torch.FloatTensor(feat), feat_rgb), self.data_label[index]

        elif "wav" in self.modality and "thr" in self.modality:
            feat = numpy.concatenate(feat, axis=0)
            feat_thr = numpy.concatenate(feat_thr, axis=0)
            feat_thr = torch.FloatTensor(feat_thr).permute(0, 3, 1, 2)

            return (torch.FloatTensor(feat), feat_thr), self.data_label[index]

        elif "wav" in self.modality:
            if self.noisy_train:
                noisy_feat = []
                for i in range(len(feat)):
                    data = noise_train_set(self.modality, feat[i], None, None, self.musan_path, False, self.p_noise, self.snr)
                    noisy_feat.append(data[0])
                noisy_feat = numpy.concatenate(noisy_feat, axis=0)
                return (torch.FloatTensor(noisy_feat)), self.data_label[index]
            else:
                feat = numpy.concatenate(feat, axis=0)
                return torch.FloatTensor(feat), self.data_label[index]

        elif "rgb" in self.modality:
            if self.noisy_train:
                noisy_feat_rgb = []
                for i in range(len(feat_rgb)):
                    data = noise_train_set(self.modality, None, torch.FloatTensor(feat_rgb[i]).permute(0, 3, 1, 2), None, self.musan_path, False, self.p_noise, self.snr)
                    noisy_feat_rgb.append(data[1])
                return torch.cat(noisy_feat_rgb, axis=0), self.data_label[index]
            else:
                feat_rgb = numpy.concatenate(feat_rgb, axis=0)
                feat_rgb = torch.FloatTensor(feat_rgb).permute(0, 3, 1, 2)
                return feat_rgb, self.data_label[index]

        elif "thr" in self.modality:
            if self.noisy_train:
                noisy_feat_thr = []
                for i in range(len(feat_thr)):
                    data = noise_train_set(self.modality, None, None, torch.FloatTensor(feat_thr[i]).permute(0, 3, 1, 2), self.musan_path, False, self.p_noise, self.snr)
                    noisy_feat_thr.append(data[2])
                return torch.cat(noisy_feat_thr, axis=0), self.data_label[index]
            else:
                feat_thr = numpy.concatenate(feat_thr, axis=0)
                feat_thr = torch.FloatTensor(feat_thr).permute(0, 3, 1, 2)
                return feat_thr, self.data_label[index]

        else:
            print("Incorrect data stream! Terminating")
            exit()

    def __len__(self):
        return len(self.data_list)


def load_test_lists(path):
    #
    # Parse 2 space separated TXT lists
    #
    with open(os.path.join(path, "rgb_list.txt")) as f:
        rgb_list = [line.split() for line in f.readlines()]

    with open(os.path.join(path, "thr_list.txt")) as f:
        thr_list = [line.split() for line in f.readlines()]

    return rgb_list, thr_list

def load_noise_data(path):
    print("Loading precomputed noise data parameters for the test list")
    noise_data = []
    with open(os.path.join(path, "noise.txt")) as f:
        for line in f.readlines():
            row = line.split()
            noise_data.append([float(row[0]), int(row[1]), int(row[2]), row[3], float(row[4])])
    return noise_data

def generate_noise_data(test_list, modality, path, p_noise, musan_path):
    os.makedirs(path)
    noise_data= []
    print("Generating noise data parameters for the test list")
    for i in range(len(test_list)):
        n = numpy.random.rand()
        if n < p_noise:
            t = numpy.random.randint(3) + 1
            if "wav" in modality and "rgb" in modality and "thr" in modality:
                mode = numpy.random.randint(3)
            elif "wav" in modality and "rgb" in modality:
                mode = numpy.random.randint(2)
            elif modality == "wav":
                mode = 0
            elif modality == "rgb":
                mode = 1
            elif modality == "thr":
                mode = 2
            if mode == 0:
                noise_t = ['music', 'noise', 'speech'][t-1]
                musan_filepath = os.path.join(musan_path, noise_t, "*", "*", "*.wav")
                selected_noises = glob.glob(musan_filepath)
                random_choice = numpy.random.randint(len(selected_noises))
                t_audio = selected_noises[random_choice]
                noise_data.append([n, t, mode, t_audio, -1])

            elif mode == 1 and (t == 1 or t ==2):
                direction = numpy.random.uniform(-1, 1)
                noise_data.append([n, t, mode, -1, direction])
            else:
                noise_data.append([n, t, mode, -1, -1])
        else:
            noise_data.append([n, -1, -1, -1, -1])

    with open(os.path.join(path, 'noise.txt'), 'w') as f:
        for line in noise_data:
            f.write("{} {} {} {} {}\n".format(line[0], line[1], line[2], line[3], line[4]))
    return noise_data

class test_dataset_loader(Dataset):
    def __init__(self, test_list, test_path, eval_frames, eval_lists_save_path, noisy_eval_lists_save_path, **kwargs):
        self.max_frames = eval_frames;
        self.num_eval = kwargs["num_eval"]
        self.test_path = test_path
        self.test_list = test_list
        self.test_list_rgb = []
        self.test_list_thr = []
        self.modality = kwargs["modality"].lower()
        self.num_images = kwargs["num_images"]
        self.img_size = (kwargs["image_width"], kwargs["image_height"])
        
        # for SF only
        self.mean_rgb = [92.19873, 70.86596, 62.344334]
        
        # for VC1 only
        #self.mean_rgb = [138.2202, 96.3156, 79.2870]

        self.mean_thr = [241.39543, 189.38396, 76.27353]
        self.musan_path = kwargs["musan_path"]
        self.noisy_eval = kwargs["noisy_eval"]
        self.p_noise = kwargs["p_noise"]
        self.snr = kwargs["snr"]
        print("\nInitializing the test_data_loader")

        num = self.num_eval

        # if rgb and thr lists exist, load them
        eval_lists_save_path = os.path.join(eval_lists_save_path, "num_images_" + str(num))
        if os.path.exists(eval_lists_save_path):
            self.test_list_rgb, self.test_list_thr = load_test_lists(eval_lists_save_path)
        else:
            # based on test_list create test_list_rgb and test_list_thr to contain the relevant names
            for audio_filename in self.test_list:
                # rgb
                clip_id = str(int(audio_filename.split('/')[-1].split('.')[0]))
                path_rgb = os.path.join(self.test_path, os.sep.join(audio_filename.split(os.sep)[:-2]), "rgb", clip_id, "*.jpg")
                img_list = get_img_list(path_rgb, num)
                self.test_list_rgb.append(img_list)
                
                path_thr = os.path.join(self.test_path, os.sep.join(audio_filename.split(os.sep)[:-2]), "thr", clip_id, "*.jpg")
                img_list = get_img_list(path_thr, num)
                self.test_list_thr.append(img_list)
                
            # save lists for further usage
            os.makedirs(eval_lists_save_path)
            
            with open(os.path.join(eval_lists_save_path, 'rgb_list.txt'), 'w') as f:
                for line in self.test_list_rgb:
                    f.write(" ".join(line) + "\n")

            with open(os.path.join(eval_lists_save_path, 'thr_list.txt'), 'w') as f:
                for line in self.test_list_thr:
                    f.write(" ".join(line) + "\n")

            
        if self.noisy_eval:
            noisy_eval_lists_save_path = os.path.join(noisy_eval_lists_save_path, self.modality)
            if os.path.exists(noisy_eval_lists_save_path):
                self.noise_data = load_noise_data(noisy_eval_lists_save_path)
            else:
                self.noise_data = generate_noise_data(self.test_list, self.modality, noisy_eval_lists_save_path, self.p_noise, self.musan_path)



    def __getitem__(self, index):
        if "wav" in self.modality:
            audio = loadWAV(os.path.join(self.test_path, self.test_list[index]), self.max_frames,
                            evalmode=True, num_eval=self.num_eval)
        if "rgb" in self.modality:
            rgb = loadIMG([os.path.join(self.test_path, f) for f in self.test_list_rgb[index]], self.img_size,
                          mean_train=self.mean_rgb)
            rgb = torch.FloatTensor(rgb).permute(0, 3, 1, 2)

        if "thr" in self.modality:
            thr = loadIMG([os.path.join(self.test_path, f) for f in self.test_list_thr[index]], self.img_size,
                          mean_train=self.mean_thr)

            thr = torch.FloatTensor(thr).permute(0, 3, 1, 2)


        if "wav" in self.modality and "rgb" in self.modality and "thr" in self.modality:
            if self.noisy_eval:
                data = noise_evaluation_set(self.modality, self.noise_data[index], audio, rgb, thr, self.musan_path, True, self.p_noise, self.snr)
                return (torch.FloatTensor(data[0]), data[1], data[2]), self.test_list[index]
            else:
                return (torch.FloatTensor(audio), rgb, thr), self.test_list[index]

        elif "wav" in self.modality and "rgb" in self.modality:
            if self.noisy_eval:
                data = noise_evaluation_set(self.modality, self.noise_data[index], audio, rgb, None, self.musan_path, True, self.p_noise, self.snr)
                return (torch.FloatTensor(data[0]), data[1]), self.test_list[index]
            else:
                return (torch.FloatTensor(audio), rgb), self.test_list[index]

        elif "wav" in self.modality and "thr" in self.modality:
            return (torch.FloatTensor(audio), thr), self.test_list[index]

        elif "wav" in self.modality:
            if self.noisy_eval:
                data = noise_evaluation_set(self.modality, self.noise_data[index], audio, None, None, self.musan_path, True, self.p_noise, self.snr)
                return torch.FloatTensor(data[0]), self.test_list[index]
            else:
                return torch.FloatTensor(audio), self.test_list[index]

        elif "rgb" in self.modality:
            if self.noisy_eval:
                data = noise_evaluation_set(self.modality, self.noise_data[index], None, rgb,  None, self.musan_path, True, self.p_noise, self.snr)
                return data[1], self.test_list[index]
            else:
                return rgb, self.test_list[index]

        elif "thr" in self.modality:
            if self.noisy_eval:
                data = noise_evaluation_set(self.modality, self.noise_data[index], None, None, thr, self.musan_path, True, self.p_noise, self.snr)
                return data[2], self.test_list[index]
            else:
                return thr, self.test_list[index]
        else:
            print("Incorrect data stream! Terminating")
            exit()


    def __len__(self):
        return len(self.test_list)


class train_dataset_sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size, distributed, seed, **kwargs):

        self.data_label = data_source.data_label;
        self.nPerSpeaker = nPerSpeaker;
        self.max_seg_per_spk = max_seg_per_spk;
        self.batch_size = batch_size;
        self.epoch = 0;
        self.seed = seed;
        self.distributed = distributed

    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_label), generator=g).tolist()

        data_dict = {}

        # Sort into dictionary of file indices for each ID
        for index in indices:
            speaker_label = self.data_label[index]
            if not (speaker_label in data_dict):
                data_dict[speaker_label] = [];
            data_dict[speaker_label].append(index);

        ## Group file indices for each class
        dictkeys = list(data_dict.keys());
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        for findex, key in enumerate(dictkeys):
            data = data_dict[key]
            numSeg = round_down(min(len(data), self.max_seg_per_spk), self.nPerSpeaker)

            rp = lol(numpy.arange(numSeg), self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Mix data in random order
        mixid = torch.randperm(len(flattened_label), generator=g).tolist()
        mixlabel = []
        mixmap = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = round_down(len(mixlabel), self.batch_size)
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        mixed_list = [flattened_list[i] for i in mixmap]

        ## Divide data to each GPU
        if self.distributed:
            total_size  = round_down(len(mixed_list), self.batch_size * dist.get_world_size()) 
            start_index = int ( ( dist.get_rank()     ) / dist.get_world_size() * total_size )
            end_index   = int ( ( dist.get_rank() + 1 ) / dist.get_world_size() * total_size )
            self.num_samples = end_index - start_index
#             sys.stdout.write("\nDividing data to each GPU.\n ID {:d} out of {:d} \n Index {:d} out of {:d}".format(dist.get_rank(), dist.get_world_size(), start_index, end_index));

            return iter(mixed_list[start_index:end_index])
        
        else:
            total_size = round_down(len(mixed_list), self.batch_size)
            self.num_samples = total_size
            return iter(mixed_list[:total_size])
        
    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, time, os, argparse, socket
import yaml
import numpy
import pdb
import torch
import glob
import zipfile
import datetime
from tuneThreshold import *
from SpeakerNet import *
from DatasetLoader import *
import torch.distributed as dist
import torch.multiprocessing as mp

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====
import os

parser = argparse.ArgumentParser(description="SpeakerNet");

parser.add_argument('--config', type=str, default=None, help='Config YAML file');

# Image input parameters
parser.add_argument('--num_images', type=int, default=1,
                    help='Number of images to extract from each recording (for both visual and thermal streams)');
parser.add_argument('--image_width', type=int, default=124,
                    help='Width of thermal and rgb images');
parser.add_argument('--image_height', type=int, default=124,
                    help='Height of thermal and rgb images');

## Data loader
parser.add_argument('--max_frames', type=int, default=200, 
                    help='Input length to the network for training');
parser.add_argument('--eval_frames', type=int, default=300,
                    help='Input length to the network for testing; 0 uses the whole files');
parser.add_argument('--batch_size', type=int, default=200, help='Batch size, number of speakers per batch');
parser.add_argument('--max_seg_per_spk', type=int, default=500,
                    help='Maximum number of utterances per speaker per epoch');
parser.add_argument('--nDataLoaderThread', type=int, default=0, help='Number of loader threads');
parser.add_argument('--seed', type=int, default=10, help='Seed for the random number generator');

## Training details
parser.add_argument('--test_interval', type=int, default=10, help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch', type=int, default=500, help='Maximum number of epochs');
parser.add_argument('--trainfunc', type=str, default="", help='Loss function');

## Optimizer
parser.add_argument('--optimizer', type=str, default="adam", help='sgd or adam');
parser.add_argument('--scheduler', type=str, default="steplr", help='Learning rate scheduler');
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate');
parser.add_argument("--lr_decay", type=float, default=0.95, help='Learning rate decay every [test_interval] epochs');
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay in the optimizer');

## Loss functions
parser.add_argument("--hard_prob", type=float, default=0.5,
                    help='Hard negative mining probability, otherwise random, only for some loss functions');
parser.add_argument("--hard_rank", type=int, default=10,
                    help='Hard negative mining rank in the batch, only for some loss functions');
parser.add_argument('--margin', type=float, default=0.1, help='Loss margin, only for some loss functions');
parser.add_argument('--scale', type=float, default=30, help='Loss scale, only for some loss functions');
parser.add_argument('--nPerSpeaker', type=int, default=1,
                    help='Number of utterances per speaker per batch, only for metric learning based losses');
parser.add_argument('--nClasses', type=int, default=5994,
                    help='Number of speakers in the softmax layer, only for softmax-based losses');

## Load and save
parser.add_argument('--initial_model', type=int, default=-1, help='Initial model weights');
parser.add_argument('--save_path', type=str, default="exps/exp1", help='Path for model and logs');
parser.add_argument('--train_lists_save_path', type=str, default="data/metadata/train",
                    help="Path to the list of filenames (train set)");
parser.add_argument('--eval_lists_save_path', type=str, default="data/metadata/", help="Path to the list of filenames (test or valid set");
parser.add_argument('--noisy_eval_lists_save_path', type=str, default="data/metadata/", help="Path to the list of noise applied to every instance of eval list (test or valid set");

## Training and test data
parser.add_argument('--train_list', type=str, default="data/metadata/train_list.txt", help='Train list');
parser.add_argument('--test_list', type=str, default="data/metadata/valid_list.txt", help='Evaluation list');
parser.add_argument('--train_path', type=str, default="data/train", help='Absolute path to the train set');
parser.add_argument('--test_path', type=str, default="data/valid", help='Absolute path to the test set');
parser.add_argument('--musan_path', type=str, default="data/musan_split", help='Absolute path to the test set');

## Model definition
parser.add_argument('--n_mels', type=int, default=40, help='Number of mel filterbanks');
parser.add_argument('--log_input', type=bool, default=False, help='Log input features')
parser.add_argument('--model', type=str, default="", help='Name of model definition');
parser.add_argument('--encoder_type', type=str, default="SAP", help='Type of encoder');
parser.add_argument('--nOut', type=int, default=512, help='Embedding size in the last FC layer');
parser.add_argument('--filters', nargs=4, type=int, default=[16, 32, 64, 128],
                    help="the list of number of filters for each of the 4 layers in ResNet34")
parser.add_argument('--modality', type=str, default="rgb",
                    help='Data streams to use, e.g. audio: "wav", visual: "rgb", thermal: "thr", all streams: "wavrgbthr');

## For test evaluation only
parser.add_argument('--eval', type=bool, default=False, dest='eval', help='Eval only')
parser.add_argument('--valid_model', type=bool, default=False,
                    help="True if you want to choose evaluate based on the performance on validation set, False otherwise (the model at the last iteration is chosen)")
parser.add_argument('--num_eval', type=int, default=10, dest='num_eval',
                    help='The number of partitions for an audio file at the evalulation mode')

## For noisy data
parser.add_argument('--noisy_eval', type=str, default=False,
                    help='If True then noisy evaluation');
parser.add_argument('--noisy_train', type=str, default=False,
                    help='If True then training with augmentations');
parser.add_argument('--p_noise', type=float, default=0.3,
                    help='The noisy probability');
parser.add_argument('--snr', type=float, default=0,
                    help='The signal to noise ratio');

## Distributed and mixed precision training
parser.add_argument('--port', type=str, default="8888", help='Port for distributed training, input as text');
parser.add_argument('--distributed', dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec', dest='mixedprec', action='store_true', help='Enable mixed precision training')
parser.add_argument('--gpu_id', type=str, default=0, help="gpu_id")

#Continue training with certain state
parser.add_argument('--load_weight', dest='load_weight', help='Load weights')


## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
            return opt.type
    raise ValueError

import math

## To get size of the model
def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])    

## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    print(gpu)
    ## Load models
    
    s = SpeakerNet(**vars(args));
    num_params = 0
    for param in s.parameters():
        num_params += param.numel()
    print('[Network %s] Total number of parameters : %.3f M' % ('ResNet', num_params / 1e6))
    with open(args.result_save_path + '/parameters.txt', 'w') as f:
        f.write("%s\n" % num_params)    
    
    
    
    print('-----------------------------------------------')        

    mem_params = sum([param.nelement()*param.element_size() for param in s.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in s.buffers()])
    mem = mem_params + mem_bufs # in bytes
    print(convert_size(mem))
    
    print('-----------------------------------------------')        

    
    if args.distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port
        
        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)
        
        s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=True)
        
        print('Loaded the model on GPU {:d}'.format(args.gpu))

    else:
        s = WrappedModel(s).cuda(args.gpu)

    it = 1
    print("Iter " + str(it))

    ## Write args to scorefile
    if args.eval:
        scorefile = open(args.result_save_path + "/scores.txt", "r");
        lines_veer = [[int(line.split()[1]), float(line.split()[3])] for line in scorefile if "VEER" in line]
        lines_veer = numpy.array(lines_veer)

        if args.valid_model:
            args.model_it = lines_veer[:, 1].argmin()
        else:
            args.model_it = -1
    else:
        scorefile = open(args.result_save_path + "/scores.txt", "a+");
        args.model_it = -1

        ## Initialise trainer and data loader
        train_dataset = train_dataset_loader(**vars(args))

        train_sampler = train_dataset_sampler(train_dataset, **vars(args))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.nDataLoaderThread,
            sampler=train_sampler,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )

    trainer = ModelTrainer(s, **vars(args))
    
    ## Load model weights
    
    modelfiles = glob.glob('%s/model0*.model' % args.model_save_path)
    modelfiles.sort()
    if len(modelfiles) >= 1 and args.initial_model == -1:

        ## Madina: enable an option to load a specific model
        if (args.model_it == -1):
            print("Requested to load the model at the latest iteration")
        else:
            print("Requested to load the model at the iteration id = {}".format(int(lines_veer[args.model_it, 0])))
        trainer.loadParameters(modelfiles[args.model_it]);
        print("Loaded model {}".format(modelfiles[args.model_it]))
        it = int(os.path.splitext(os.path.basename(modelfiles[args.model_it]))[0][5:]) + 1

    elif (args.initial_model != -1):
        print('TRANSFER LEARNING. Initializing initial weights...')
#         initial_weight_path = '/'.join(args.model_save_path.split('/')[0:-2]) + f'/exp{args.initial_model}/model'
#         initialmodelfiles = glob.glob('%s/model0*.model' % initial_weight_path)
#         initialmodelfiles.sort()
        
#         breakpoint()

#         initial_result_path = '/'.join(args.result_save_path.split('/')[0:-2]) + f'/exp{args.initial_model}/result'
#         initialscorefile = open(initial_result_path + "/scores.txt", "r");
#         initiallines_veer = [[int(line.split()[1]), float(line.split()[3])] for line in initialscorefile if "VEER" in line]
#         initiallines_veer = numpy.array(initiallines_veer)
#         breakpoint()

#         id_to_load = initiallines_veer[:, 1].argmin()
#         print(f"Requested to load the model at the iteration id = {id_to_load+1}")
        print(f"Requested to load the model at the iteration id = {args.initial_model}")
        print(modelfiles[args.initial_model - 1])
        trainer.loadParameters(modelfiles[args.initial_model - 1]);
        print(f"Model {args.initial_model} loaded!");
        
    for ii in range(1, it):
        trainer.__scheduler__.step()
#     breakpoint()
    ## Evaluation code - must run on single GPU
    if args.eval:
        pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())

        print('Total parameters: ', pytorch_total_params)
        print('Test list', args.test_list)

        assert args.distributed == False

        sc, lab, _ = trainer.evaluateFromList(**vars(args))
        result = tuneThresholdfromScore(sc, lab, [1, 0.1]);

        recordPredictions(sc, result[4], args.result_save_path, **vars(args))

        p_target = 0.05
        c_miss = 1
        c_fa = 1

        fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
        mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa)

        scorefile.close()
        scorefile = open(args.result_save_path + "/scores.txt", "r");
        lines = [line for line in scorefile if ("Epoch {:d}".format(int(lines_veer[args.model_it, 0])) in line)]
        scorefile.close()

        eval_scorefile = open(args.result_save_path + "/eval_scores.txt", "a+");
        print(lines[0])
        eval_scorefile.write(lines[0])
        print(lines[1])
        eval_scorefile.write(lines[1])
        print('\n', time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}".format(result[1]),
              "MinDCF {:2.5f}".format(mindcf), "Noisy evaluation {}".format(args.noisy_eval), "snr {}".format(args.snr));
        eval_scorefile.write(
            "EER {:2.4f} MinDCF {:2.5f} Noisy evaluation {} snr {} \n".format(result[1], mindcf, args.noisy_eval, args.snr))
        eval_scorefile.close()
        quit();

    ## Save training code and params
    if args.gpu == 0:
        pyfiles = glob.glob('./*.py')
        strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        zipf = zipfile.ZipFile(args.result_save_path + '/run%s.zip' % strtime, 'w', zipfile.ZIP_DEFLATED)
        for file in pyfiles:
            zipf.write(file)
        zipf.close()

        with open(args.result_save_path + '/run%s.cmd' % strtime, 'w') as f:
            f.write('%s' % args)
    starttime = time.time()
    previoustime = time.time()
    ## Core training script
    for it in range(it, args.max_epoch + 1):

        train_sampler.set_epoch(it)

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]
        mem_params = sum([param.nelement()*param.element_size() for param in s.parameters()])
        mem_bufs = sum([buf.nelement()*buf.element_size() for buf in s.buffers()])
        mem = mem_params + mem_bufs # in bytes
        print(convert_size(mem))
    
        loss, traineer = trainer.train_network(train_loader, verbose=(args.gpu == 0));

        if args.gpu == 0:
            currenttime = time.time()
            print('\n', time.strftime("%Y-%m-%d %H:%M:%S"),
                  "Epoch {:d} TEER/TAcc {:2.2f} TLOSS {:f} LR {:f} Total time: {:2.2f} Time for epoch {:2.2f}".format(it, traineer, loss, max(clr), currenttime - starttime, currenttime - previoustime));
            scorefile.write("Epoch {:d} TEER/TAcc {:2.2f} TLOSS {:f} LR {:f} Total time: {:2.2f} Time for epoch {:2.2f}\n".format(it, traineer, loss, max(clr), currenttime - starttime, currenttime - previoustime));
            previoustime = currenttime
            
        if it % args.test_interval == 0:


            sc, lab, _ = trainer.evaluateFromList(**vars(args))
            if args.gpu == 0:
                
                result = tuneThresholdfromScore(sc, lab, [1, 0.1]);

                print('\n', time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d} VEER {:2.4f}".format(it, result[1]));
                scorefile.write("Epoch {:d} VEER {:2.4f}\n".format(it, result[1]));

                trainer.saveParameters(args.model_save_path + "/model%09d.model" % it);


                scorefile.flush()

        if ("nsml" in sys.modules) and args.gpu == 0:
            training_report = {};
            training_report["summary"] = True;
            training_report["epoch"] = it;
            training_report["step"] = it;
            training_report["train_loss"] = loss;

            nsml.report(**training_report);

    scorefile.close();


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====


def main(args):
    if ("nsml" in sys.modules):
        args.save_path = os.path.join(args.save_path, SESSION_NAME.replace('/', '_'))

    args.model_save_path = args.save_path + "/model"
    args.result_save_path = args.save_path + "/result"
    args.feat_save_path = ""

    if "valid" in args.test_list:
        args.eval_lists_save_path = os.path.join(args.eval_lists_save_path, "valid")
        args.noisy_eval_lists_save_path = os.path.join(args.noisy_eval_lists_save_path, "valid")
    else:
        args.eval_lists_save_path = os.path.join(args.eval_lists_save_path, "test")
        args.noisy_eval_lists_save_path = os.path.join(args.noisy_eval_lists_save_path, "test")

    if not (os.path.exists(args.model_save_path)):
        os.makedirs(args.model_save_path)

    if not (os.path.exists(args.result_save_path)):
        os.makedirs(args.result_save_path)

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:', args.save_path)

    if args.distributed:
            mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)


if __name__ == '__main__':
    args = parser.parse_args();

    if args.config is not None:
        with open(args.config, "r") as f:
            yml_config = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in yml_config.items():
            if k in args.__dict__:
                typ = find_option_type(k, parser)
                args.__dict__[k] = typ(v)
            else:
                sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))

    ## Try to import NSML
    try:
        import nsml
        from nsml import HAS_DATASET, DATASET_PATH, PARALLEL_WORLD, PARALLEL_PORTS, MY_RANK
        from nsml import NSML_NFS_OUTPUT, SESSION_NAME
    except:
        pass;

    ### To select a specific GPU available
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    ### To select a specific seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_deterministic(True)  # for pytorch version 1.7
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True) #for pytorch version 1.8

    main(args)

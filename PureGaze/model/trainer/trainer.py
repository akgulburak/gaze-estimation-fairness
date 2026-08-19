import os, sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/home/chengyihua/utils/")
import model
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import yaml
import cv2
import ctools
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import argparse

from datasets import Gaze360
from torch.utils.data import DataLoader, default_collate
from torch.utils.data import WeightedRandomSampler

def apply_class_weighted_loss(predicted, groundtruth, loss_function, weight, img, face, K):
    weight = weight / K
    loss = loss_function(predicted, img, groundtruth, face)
    weighted_loss = loss * weight.to("cuda:0")
    mean_weighted_loss = weighted_loss.mean()
    return mean_weighted_loss

def main(train):
    # Setup-----------------------------------------------------------
    #dataloader = importlib.import_module(f"reader.{config.reader}")

    torch.cuda.set_device(config.device)

    attentionmap = cv2.imread(config.map, 0)/255
    attentionmap = torch.from_numpy(attentionmap).type(torch.FloatTensor)

    data = config.data
    save = config.save
    params = config.params

    use_oversample = config.use_oversample
    use_class_weights = config.use_class_weights

    # Prepare dataset-------------------------------------------------
    #dataset = dataloader.loader(data, params.batch_size, shuffle=True, num_workers=4)
    dataset = Gaze360(data, use_oversample=use_oversample, use_class_weights=use_class_weights)
    if config.use_reweight:
        sampler = WeightedRandomSampler(dataset.weights, len(dataset.weights))
        dataset = DataLoader(dataset, batch_size=params.batch_size, sampler = sampler, num_workers=4)
    else:
        dataset = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=4)
    #

    # Build model
        # build model ------------------------------------------------
    print("===> Model building <===")
    net = model.Model(); net.train(); net.cuda()

    if config.pretrain:
        net.load_state_dict(torch.load(config.pretrain), strict=False)

    print("optimizer building")
    geloss_op = model.Gelossop(attentionmap, w1=3, w2=1)
    deloss_op = model.Delossop()

    ge_optimizer = optim.Adam(net.feature.parameters(),
             lr=params.lr, betas=(0.9,0.95))

    ga_optimizer = optim.Adam(net.gazeEs.parameters(), 
             lr=params.lr, betas=(0.9,0.95))

    de_optimizer = optim.Adam(net.deconv.parameters(), 
            lr=params.lr, betas=(0.9,0.95))

    # scheduler = optim.lr_scheduler.StepLR(optimizer,
            #step_size=params.decay_step, gamma=params.decay)

    # prepare for training ------------------------------------

    length = len(dataset);
    total = length * params.epoch

    savepath = os.path.join(args.save_metapath, save.folder, f"checkpoint")

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    timer = ctools.TimeCounter(total)

  
    print("Training")
    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
        for epoch in range(1, config["params"]["epoch"]+1):
            for i, (data, label) in enumerate(dataset):
                # Acquire data
                data["face"] = data["face"].cuda()
                label = label.cuda()
 
                # forward
                gaze, img = net(data)

                ge_optimizer.zero_grad()
                ga_optimizer.zero_grad()
                de_optimizer.zero_grad()

                for param in net.deconv.parameters():
                    param.requires_grad=False

                if config.use_class_weights:
                    geloss = apply_class_weighted_loss(gaze, label, geloss_op, data["weight"], img, data["face"], data["K_DG"])
                else:
                    geloss = geloss_op(gaze, img, label, data["face"])

                geloss.backward(retain_graph=True)


                for param in net.deconv.parameters():
                    param.requires_grad=True


                for param in net.feature.parameters():
                    param.requires_grad = False

                deloss = deloss_op(img, data["face"])
                deloss.backward()

                for param in net.feature.parameters():
                    param.requires_grad=True
 
                ge_optimizer.step()
                ga_optimizer.step()
                de_optimizer.step()
                
                rest = timer.step()/3600

                # print logs
                if i % 20 == 0:
                    log = f"[{epoch}/{params.epoch}]: " + \
                          f"[{i}/{length}] " +\
                          f"gloss:{geloss} " +\
                          f"dloss:{deloss} " +\
                          f"lr:{params.lr} " +\
                          f"rest time:{rest:.2f}h"

                    print(log)
                    outfile.write(log + "\n")
                    sys.stdout.flush()   
                    outfile.flush()

            if epoch % config["save"]["step"] == 0:
                torch.save(net.state_dict(), os.path.join(savepath, f"Iter_{epoch}_{save.name}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    parser.add_argument('-c', '--config', type=str,
                        help='Path to the config file.')
    parser.add_argument('-s', '--save_metapath', type=str,
                        help='Path to the config file.')

    args = parser.parse_args()

    config = edict(yaml.load(open(args.config), Loader=yaml.FullLoader))

    main(config)
 

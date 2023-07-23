from training_utils import loss, logger, metrics
import torch.nn as nn
import os.path as osp
import os
from tqdm import tqdm
import cv2
import numpy as np
import time


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(args, epoch, model, loader, dataset_size, optimizer, device, log_path):
    model.train()
    segloss = loss.segmentation_loss()
    epoch_data = {}
    for k in logger.PRESET_SEGMENTATION_LOGS:
        epoch_data[k] = 0
    epoch_data["epochs"] = epoch
    epoch_data["lr"] = get_lr(optimizer)
    epoch_data["num_samples"] = dataset_size

    print("==========================================================================")
    print("Training models: Epoch:", epoch)

    if osp.isfile(osp.join(log_path, args.train_model+"_train_logs.json")):
        log = logger.get_log(
            osp.join(log_path, args.train_model+"_train_logs.json"))
    else:
        log = logger.create_log(
            osp.join(log_path, args.train_model+"_train_logs.json"), mode="segmentation")
    dataset_iterator = tqdm(enumerate(loader), total=len(loader))

    Ti = int(round(time.time() * 1000))
    for batch_idx, pack in dataset_iterator:
        img, map = pack["img"].to(device), pack["map"].to(device)
        pred = model(img)
        optimizer.zero_grad()
        loss = segloss(map, pred, args.loss_weights)
        loss.backward()
        optimizer.step()

        epoch_data["loss"] += loss.item()/dataset_size
        mets = metrics.segmentation_metrics(pred, map)
        for k in mets.keys():
            epoch_data[k] += mets[k]/dataset_size
    Tf = int(round(time.time() * 1000))
    epoch_data["delta-time"] = (Tf - Ti)/dataset_size

    logger.log_epoch(log, epoch_data)
    logger.save_log(log, osp.join(
        log_path, args.train_model+"_train_logs.json"))
    print("train_iou:", epoch_data["iou"])
    print("==========================================================================")
    print("==========================================================================")

    return epoch_data


def test_one_epoch(args, epoch, model, loader, dataset_size, device, log_path):
    model.eval()
    segloss = loss.segmentation_loss()
    epoch_data = {}
    for k in logger.PRESET_SEGMENTATION_LOGS:
        epoch_data[k] = 0
    epoch_data["epochs"] = epoch
    epoch_data["lr"] = "N/A"
    epoch_data["num_samples"] = dataset_size

    print("==========================================================================")
    print("Testing model for Epoch:", epoch)

    if osp.isfile(osp.join(log_path, args.train_model+"_test_logs.json")):
        log = logger.get_log(
            osp.join(log_path, args.train_model+"_test_logs.json"))
    else:
        log = logger.create_log(
            osp.join(log_path, args.train_model+"_test_logs.json"), mode="segmentation")
    dataset_iterator = tqdm(enumerate(loader), total=len(loader))

    Ti = int(round(time.time() * 1000))
    for batch_idx, pack in dataset_iterator:
        img, map = pack["img"].to(device), pack["map"].to(device)
        pred = model(img)
        loss = segloss(map, pred, args.loss_weights)

        epoch_data["loss"] += loss.item()/dataset_size
        mets = metrics.segmentation_metrics(pred, map)
        for k in mets.keys():
            epoch_data[k] += mets[k]/dataset_size
    Tf = int(round(time.time() * 1000))
    epoch_data["delta-time"] = (Tf - Ti)/dataset_size

    logger.log_epoch(log, epoch_data)
    logger.save_log(log, osp.join(
        log_path, args.train_model+"_test_logs.json"))
    print("test_iou:", epoch_data["iou"])
    print("==========================================================================")
    print("==========================================================================")

    return epoch_data

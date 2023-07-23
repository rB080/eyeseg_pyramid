from net import pynet
from loaders import load_ref
import torch
from training_utils import utils as SEG
import os.path as osp


def run(args):
    device = torch.device(args.device)
    print("train device:", device)
    model = pynet.get_model(args)

    _, trainset_size, train_loader = load_ref.get_loader(args, split="train")
    _, testset_size, test_loader = load_ref.get_loader(args, split="test")

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.seg_lr, weight_decay=0.0001)

    for epoch in range(1, args.seg_epochs+1):
        D = SEG.train_one_epoch(
            args, epoch, model, train_loader, trainset_size, optimizer, device, osp.join(args.base, args.workspace, "logs"))
        D = SEG.test_one_epoch(
            args, epoch, model, test_loader, testset_size, device, osp.join(args.base, args.workspace, "logs"))
        torch.save(model.state_dict(), osp.join(
            args.base, args.workspace, "saved_models", args.train_model+".pth"))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

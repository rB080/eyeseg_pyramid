import argparse
import os
import os.path as osp
from engine import train, generate, analyze
from training_utils.logger import Logger
import numpy as np
import json


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


defaults = {
    "base": "/lustre07/scratch/rb080/work/Outputs",
    "data_root": "/lustre07/scratch/rb080/work/Data/JSRT_dataset",
    "segdata_root": "/lustre07/scratch/rb080/work/Data/segmentation_data"
}


def get_list(s):
    L = s.split(",")
    for l in L:
        l = float(l)
    return L


def get_args_parser():
    parser = argparse.ArgumentParser()

    # Training settings
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--dataparallel', default=True, type=str2bool)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--batch_size', default=64, type=int)

    #Workspace and Paths
    parser.add_argument('--workspace', default='untitled_training', type=str)
    parser.add_argument('--log_name', default='new_log', type=str)
    parser.add_argument('--base', default=defaults["base"], type=str)

    # Jobs
    parser.add_argument('--train', default=False, type=str2bool)
    parser.add_argument('--generate', default=False, type=str2bool)
    parser.add_argument('--analyze', default=False, type=str2bool)

    # Train_parameters
    parser.add_argument('--train_data_root',
                        default=defaults["data_root"], type=str)
    parser.add_argument('--train_dataset', default="refuge", type=str)
    parser.add_argument('--train_model', default="pynet", type=str)
    parser.add_argument('--num_epochs', default=500, type=int)
    parser.add_argument('--loss_weights', default="0.3,0.5,0.1,0.1", type=str)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--lr_red_type', default="discrete", type=str)
    parser.add_argument('--lr_red_factor', default=10, type=float)
    # Discrete Reduction Parameters
    parser.add_argument('--lr_red_points', default="50,75", type=str)
    # Continuous Reduction Parameters
    parser.add_argument('--lr_red_range', default=100, type=int)

    # Generate_parameters
    parser.add_argument('--gen_data_root',
                        default=defaults["data_root"], type=str)
    parser.add_argument('--gen_dataset', default="refuge", type=str)
    parser.add_argument('--gen_model', default="pynet", type=str)
    parser.add_argument('--gen_load_model', default="pynet1.pth", type=str)

    # Analyze_parameters
    parser.add_argument(
        '--an_data_root', default=defaults["data_root"], type=str)
    parser.add_argument('--an_dataset', default="refuge", type=str)
    parser.add_argument('--an_model', default="pynet", type=str)
    parser.add_argument('--an_load_model', default="pynet1.pth", type=str)

    args = parser.parse_args()

    args.loss_weights = get_list(args.loss_weights)
    args.lr_red_points = np.array(get_list(args.lr_red_points))
    args.lr_red_points = (
        (args.lr_red_points - args.lr_red_points.min())/args.lr_red_points.max()).tolist()

    # Assertions
    # make other datasets available
    assert args.dataset in ["refuge"], "Dataset currently unavailable"
    assert args.lr_red_type in ["discrete",
                                "continuous"], "Undefined reduction method"
    assert args.model in ["pynet"], "Unavailable model"

    return args


args = get_args_parser()
configs = vars(args)

# Workspace creation:
workbase = osp.join(args.base, args.workspace)
log_base = osp.join(workbase, "logs")
mod_base = osp.join(workbase, "saved_models")
analysis_base = osp.join(workbase, "analyses")
out_base = osp.join(workbase, "outputs", args.dataset)

if osp.isdir(workbase) == False:
    os.makedirs(workbase)
if osp.isdir(log_base) == False:
    os.makedirs(log_base)
if osp.isdir(mod_base) == False:
    os.makedirs(mod_base)
if osp.isdir(analysis_base) == False:
    os.makedirs(analysis_base)
if osp.isdir(out_base) == False:
    os.makedirs(out_base)
print("Workspace Created Successfully!", flush=True)

Logger(osp.join(workbase, args.log_name) + ".log")

# Show console settings:
print("Console Settings:")
print(configs)
file = open(os.path.join(workbase, 'configs.json'), 'w')
file.write(json.dumps(configs))
file.close()
print("Console settings saved!")

# Job lists:
if args.train:  # Segmentation model training
    print("Training Segmentation Model now!")
    train.run(args)
    print("Segmentation Model Training Done!!")

if args.generate:  # Save segmentation maps from trained model
    print("Saving segmentation maps from pretrained segmentation model now!")
    generate.run(args)
    print("Segmentation maps saved successfully!!")

if args.analyze:  # Make analyses for the trainings
    print("Starting Analysis now!")
    analyze.run(args)
    print("Analysis preparation Complete!!")

import argparse
import copy
import os
from collections import OrderedDict
import numpy as np

import train

os.environ['HF_HOME'] = './'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
access_token = "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"
os.environ['HF_TOKEN'] = access_token
from resnet_attack_todo import ResnetPGDAttacker, ResNetNormalization
from datasets import load_dataset
from torchvision.models import resnet50, ResNet50_Weights
from functools import partial
from torchvision.models import resnet50
from resnet_attack_todo import ImageClassification
from torch.utils.data import DataLoader
import torch
from matplotlib import pyplot as plt
import pandas as pd
from trainer import Trainer
from dataset import AdvDataset

parser = argparse.ArgumentParser(description="Attacking a Resnet50 model")
parser.add_argument('--eps', type=float, help='maximum perturbation for PGD attack', default=8 / 255)
parser.add_argument('--alpha', type=float, help='step size for PGD attack', default=2 / 255)
parser.add_argument('--steps', type=int, help='number of steps for PGD attack', default=20)
parser.add_argument('--batch_size', type=int, help='batch size for PGD attack', default=100)
parser.add_argument('--batch_num', type=int, help='number of batches on which to run PGD attack', default=20)
parser.add_argument('--results', type=str, help='name of the file to save the results to', default="adv_model.pt")
parser.add_argument('--resultsdir', type=str, help='name of the folder to save the results to', default='ckpt/')
parser.add_argument('--seed', type=int, help='set manual seed value for reproducibility, default 1234',
                    default=1234)
parser.add_argument("--adv_data_path", type=str, help="path to adversarial dataset", default=None)
parser.add_argument("--val_steps", type=int, help="steps to start evaluation", default=200)
parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
parser.add_argument("--epochs", type=int, help="number of epochs", default=3)
args = parser.parse_args()

RESULTS_DIR = args.resultsdir
RESULTS_PATH = os.path.join(RESULTS_DIR, args.results)
if not os.path.isdir(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

if args.seed:
    SEED = args.seed
    torch.manual_seed(SEED)
else:
    SEED = torch.seed()

EPS = args.eps
ALPHA = args.alpha
STEPS = args.steps

BATCH_SIZE = args.batch_size
BATCH_NUM = args.batch_num
if BATCH_NUM is None:
    BATCH_NUM = 1281167 // BATCH_SIZE + 1
assert BATCH_NUM > 0

print('Loading model...')
# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
clean_model = resnet50(weights=weights)
adv_model = resnet50()
# adv_model.load_state_dict(torch.load("ckpt/acc0.714-epoch0-lr0.0001-step2200-mixed.pth"))
state_dict = copy.deepcopy(clean_model.state_dict())

adv_model.load_state_dict(state_dict)

# model = ResNetNormalization(model, mean=weights.transforms().mean, std=weights.transforms().std)
preprocess = weights.transforms()

# Step 2: Load and preprocess data
print('Loading data...')

# Load ImageNet-1k dataset from Huggingface



def preprocess_img(example):
    example['image'] = preprocess(example['image'])
    return example


# Process the training data
# Filter out grayscale images
if args.adv_data_path is None:
    train_ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
    train_ds = train_ds.filter(lambda example: example['image'].mode == 'RGB')
    # Preprocess function will be applied to images on-the-fly whenever they are being accessed in the loop
    train_ds = train_ds.map(preprocess_img)
    train_ds = train_ds.shuffle(seed=SEED)
    # Only take desired portion of dataset
    train_ds = train_ds.take(BATCH_NUM * BATCH_SIZE)
    train_dset_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
else:
    adv_data = torch.load(args.adv_data_path)
    adv_images = adv_data['adv_images']
    adv_labels = adv_data['labels']
    train_ds = AdvDataset(adv_images, adv_labels)
    train_dset_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_ds.collate_fn)


# dset_classes = weights.meta["categories"]
attacker = ResnetPGDAttacker(clean_model, train_dset_loader)
attacker.eps = args.eps
attacker.alpha = args.alpha
attacker.steps = args.steps
# Process val_data
val_ds = load_dataset("ILSVRC/imagenet-1k", split='validation', streaming=True)
val_ds = val_ds.filter(lambda example: example['image'].mode == 'RGB')
val_ds = val_ds.map(preprocess_img)
val_ds = val_ds.take(2000)
val_dset_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# Train modelacc0.72-epoch2-lr0.0001-step300-mixed.pth
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainer = Trainer(adv_model, attacker, train_dset_loader, val_dset_loader, device, args.lr,
                  args.epochs, attack=args.adv_data_path is None, val_steps=args.val_steps)
# trainer.val()
# print(trainer.acc)
trainer.fit()
# trainer.val()
# print(trainer.acc)
# torch.save(adv_model.state_dict(), RESULTS_PATH)
exit(0)

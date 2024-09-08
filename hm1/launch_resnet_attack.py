import argparse
import os

import numpy as np

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

parser = argparse.ArgumentParser(description="Attacking a Resnet50 model")
parser.add_argument('--eps', type=float, help='maximum perturbation for PGD attack', default=8 / 255)
parser.add_argument('--alpha', type=float, help='step size for PGD attack', default=2 / 255)
parser.add_argument('--steps', type=int, help='number of steps for PGD attack', default=20)
parser.add_argument('--batch_size', type=int, help='batch size for PGD attack', default=100)
parser.add_argument('--batch_num', type=int, help='number of batches on which to run PGD attack', default=20)
parser.add_argument('--results', type=str, help='name of the file to save the results to',  default='adv_image')
parser.add_argument('--resultsdir', type=str, help='name of the folder to save the results to', default='results')
parser.add_argument('--seed', type=int, help='set manual seed value for reproducibility, default 1234',
                    default=1234)
parser.add_argument('--test', action='store_true', help='test that code runs', default=False)
parser.add_argument("--plot", action='store_true', help="plot the results", default=False)
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
model = resnet50(weights=weights)
# model.load_state_dict(torch.load("ckpt/acc0.714-epoch0-lr0.0001-step2200-mixed.pth"))
# model = ResNetNormalization(model, mean=weights.transforms().mean, std=weights.transforms().std)
preprocess = weights.transforms()
# preprocess = ImageClassification(
#     crop_size=weights.transforms().crop_size,
#     resize_size=weights.transforms().resize_size,
#     interpolation=weights.transforms().interpolation,
#     antialias=weights.transforms().antialias,
#     mean=weights.transforms().mean,
#     std=weights.transforms().std,
# )# no normalization step
# preprocess = partial(ImageClassification, crop_size=224, resize_size=232)()
# Step 2: Load and preprocess data
print('Loading data...')

# Load ImageNet-1k dataset from Huggingface
ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)


def preprocess_img(example):
    example['image'] = preprocess(example['image'])
    return example


# Filter out grayscale images
ds = ds.filter(lambda example: example['image'].mode == 'RGB')
# Preprocess function will be applied to images on-the-fly whenever they are being accessed in the loop
ds = ds.map(preprocess_img)
ds = ds.shuffle(seed=SEED)
# Only take desired portion of dataset
ds = ds.take(BATCH_NUM * BATCH_SIZE)

dset_loader = DataLoader(ds, batch_size=BATCH_SIZE)
dset_classes = weights.meta["categories"]
attacker = ResnetPGDAttacker(model, dset_loader)

if args.plot:
    x = []
    y = []
    for eps in np.arange(0,9):
        print(f"Attack configs: eps = {eps}/255, alpha = {ALPHA}, steps = {STEPS}, batch size = {BATCH_SIZE}")
        x.append(f"{eps}/255")
        attacker = ResnetPGDAttacker(model, dset_loader)
        attacker.pgd_batch_attack(eps, ALPHA, STEPS, BATCH_NUM)
        y.append(attacker.adv_acc * 100)
    pd.to_pickle([x, y], "x-y.pkl")
    plt.plot(x, y)
    plt.title(
        f"Model accuracy on adversarial images for various epsilon values\n Alpha = 2/255, Steps = {STEPS}, Batch size = {BATCH_SIZE}, Batch num = {BATCH_NUM}")

    plt.xlabel(f"Episode values")
    plt.ylabel("Attack accuracy (%)")
    plt.savefig("result.png")
    plt.show()

ALL = "all"
if args.test:
    print(f"===Testing on {BATCH_NUM if BATCH_NUM else ALL} batches of data===")
    attacker.compute_accuracy(BATCH_NUM)
    print(f"Accuracy on original images: {attacker.acc * 100}%")
    torch.save({
        'acc': attacker.acc,
    }, RESULTS_PATH)

else:

    print(f"===Launching PGD attack on {BATCH_NUM if BATCH_NUM else ALL} batches of data===")
    print(f"Attack configs: eps = {EPS}, alpha = {ALPHA}, steps = {STEPS}, batch size = {BATCH_SIZE}")

    attacker.pgd_batch_attack(EPS, ALPHA, STEPS, BATCH_NUM)
    print(f"Accuracy on original images: {attacker.acc * 100}%")
    print(f"Accuracy on adversarial images: {attacker.adv_acc * 100}%")

    torch.save({
        'acc': attacker.acc,
        'adv_acc': attacker.adv_acc,
        'adv_images': attacker.adv_images,
        'labels': attacker.labels,
    }, RESULTS_PATH)

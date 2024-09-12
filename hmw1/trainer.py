import random

import torch
import torch.optim as optim
from tqdm import tqdm


class Trainer:
    def __init__(self, model, adv_sample_generator, train_dataloader, val_dataloader, device,
                 lr=1e-3, epochs=3, attack=False, val_steps=100):
        self.model = model.to(device)
        self.adv_sample_generator = adv_sample_generator
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = torch.nn.CrossEntropyLoss()
        # self.loss_fn = torch.nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.attack = attack
        self.val_steps = val_steps
        self.train_adv_samples_cache = []
        self.train_labels_cache = []
        self.val_adv_samples_cache = []
        self.val_labels_cache = []


    def fit(self):

        for epoch in range(self.epochs):
            self.model.train()
            step_bar = tqdm(
                self.train_dataloader,
                desc="Train step of epoch %d" % epoch,
            )
            for i, inputs in enumerate(step_bar):
                self.model.train()
                if self.attack:
                    adv_images = self.adv_sample_generator.pgd_attack(**inputs).to(self.device)
                else:
                    adv_images = inputs["image"].to(self.device)
                labels = inputs['label']
                outputs = self.model(adv_images)
                # Calculate loss
                loss = self.loss_fn(outputs, labels.to(self.device))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                step_bar.set_postfix(loss=loss.item())
                if (i+1) % self.val_steps == 0:
                    self.val()
                    print(self.acc)
                    torch.save(self.model.state_dict(), f'ckpt/acc{self.acc}-epoch{epoch}-lr{self.lr}-step{i+1}-mixed.pth')
                # step_bar.update()
            # self.val()
            # print(self.acc)

    def val(self):
        print("Start validation ...")
        self.model.eval()
        # with torch.no_grad():
        correct = 0
        total = 0

        for i, inputs in enumerate(tqdm(self.val_dataloader)):
            if i >= len(self.val_labels_cache):
                adv_images = self.adv_sample_generator.pgd_attack(**inputs)
                labels = inputs['label']
                self.val_adv_samples_cache.append(adv_images.cpu())
                self.val_labels_cache.append(labels.cpu())
            else:
                adv_images = self.val_adv_samples_cache[i]
                labels = self.val_labels_cache[i]
            # inputs = {k: v.to(self.device) for (k, v) in inputs.items()}
            with torch.no_grad():
                outputs = self.model(adv_images.to(self.device)).softmax(1)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels.to(self.device)).sum().item()
                total += predictions.size(0)
        self.acc = correct / total

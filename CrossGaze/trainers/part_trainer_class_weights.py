import torch
import numpy as np

def apply_class_weighted_loss(predicted, groundtruth, loss_function, weight, K):
    weight = weight / K
    loss = loss_function(predicted, groundtruth, )
    weighted_loss = loss * weight.to("cuda:0")
    mean_weighted_loss = weighted_loss.mean()
    return mean_weighted_loss

class PartTrainerClassWeights():
    def __init__(self, args, train_loader, device, optimizer, scheduler, criterion):
        self.args = args
        self.train_loader = train_loader
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

    def train(self, model):
        model.train()
        total_train_loss = 0
        K_DG = self.train_loader.dataset.K_DG

        counter = 0 # DELETE
        for i, data in enumerate(self.train_loader):
            counter += 1 # DELETE
            if counter>10: # DELETE
                exit() # DELETE
            images = data['img']
            labels = data['label3d'].to(self.device)
            weight = data['weight']

            face_images = images[0].to(self.device)
            left_eye_images = images[1].to(self.device)
            right_eye_images = images[2].to(self.device)

            self.optimizer.zero_grad()
            pred = model(face_images, left_eye_images, right_eye_images)

            loss = apply_class_weighted_loss(pred, labels, self.criterion, weight, K_DG)
            
            loss.backward()
            self.optimizer.step()

            total_train_loss += loss.item() * 100
            if self.args.scheduler == 'one-cycle':
                self.scheduler.step()

        if self.args.scheduler != 'one-cycle':
            self.scheduler.step()

        train_loss = np.round(total_train_loss / len(self.train_loader), 2)
        print("TRAIN LOSS: " + str(train_loss))

        return train_loss

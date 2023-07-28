#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from dataloader import *
from models.DeepPCO import *

import torchsummary
from tqdm import tqdm

# define model hyperparmeters
hyperparams = {'Epoch': 30,
               'lr': 1e-4,
               'betas': [0.9, 0.999],
               'batch_size': 8,
               'wd': 1e-5,
               'step_size': 10,
               'gamma': 0.5}


def calculate_rmse(predictions, targets):
    t_mse = nn.MSELoss()(predictions[:3], targets[:3])
    q_mse = nn.MSELoss()(predictions[3:], targets[3:])
    t_rmse, q_rmse = torch.sqrt(t_mse), torch.sqrt(q_mse)
    return t_rmse, q_rmse


def train_one_epoch(epoch, train_loader):
    train_loss = 0.0
    translation_acc = 0.0
    orientation_acc = 0.0

    model.train()
    progress_bar = tqdm(train_loader, total=len(train_loader),
                        desc=f'Epoch {epoch}/{num_epochs}, Train Loss: 0.0000')

    for batch_idx, (img, gt) in enumerate(progress_bar):
        img, gt = img.to(device), gt.to(device)
        output = model(img)  # output is (x, y, z, roll, pitch, yaw)
        loss = criterion(output, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(f"output: {output[0]}\ngt: {gt[0]}")

        train_t_acc, train_q_acc = calculate_rmse(output, gt)

        train_loss += loss.item()
        translation_acc += train_t_acc.item()
        orientation_acc += train_q_acc.item()

        progress_bar.set_description(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss / (batch_idx + 1):.4f}, Train translation acc: {train_t_acc:.4f}, Train orientation acc: {train_q_acc:.4f}')
    
    print(f"Epoch: {epoch}, Train loss: {train_loss / len(train_loader):.4f}, Train translation acc: {train_t_acc:.4f}, Train orientation acc: {train_q_acc:.4f}")

    train_loss /= len(train_loader)
    translation_acc /= len(train_loader)
    orientation_acc /= len(train_loader)

    progress_bar.close()
    return train_loss, translation_acc, orientation_acc


def test_epoch(test_loader):
    model.eval()
    test_loss = 0.0
    translation_acc = 0.0
    orientation_acc = 0.0

    with torch.no_grad():
        for batch_idx, (img, gt) in enumerate(test_loader):
            img, gt = img.to(device), gt.to(device)
            output = model(img)
            loss = criterion(output, gt)
            test_t_acc, test_q_acc = calculate_rmse(output, gt)

            test_loss += loss.item()
            translation_acc += test_t_acc.item()
            orientation_acc += test_q_acc.item()

    print(f"Epoch: {epoch}, Test Loss: {test_loss/ len(test_loader):.4f}, Test translation acc: {test_t_acc:.4f}, Test orientaion acc: {test_q_acc:.4f}\n")

    test_loss /= len(test_loader)
    translation_acc /= len(test_loader)
    orientation_acc /= len(test_loader)

    return test_loss, translation_acc, orientation_acc


def adjust_weight_decay(optimizer, epoch, init_weight_decay=hyperparams['wd'], decay_factor=hyperparams['gamma'], decay_epochs=hyperparams['step_size']):
    if epoch > 0 and epoch % decay_epochs == 0:
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] *= decay_factor
    
    else:
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = init_weight_decay


if __name__ == '__main__':
    root_dir = '/home/smeet/catkin_ws/src/PointFlow-Odometry/dataset/custom_sequence/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepPCO().to(device)
    criterion = Criterion(orientation='euler', k=100.0).to(device)
    optimizer = optim.Adam(model.parameters(),
                           betas=hyperparams['betas'],
                           lr=hyperparams['lr'],
                           weight_decay=hyperparams['wd'])

    num_epochs = hyperparams['Epoch']
    lr_scheduler = StepLR(optimizer, step_size=hyperparams['step_size'], gamma=hyperparams['gamma'], verbose=True)

    writer = SummaryWriter()
    torchsummary.summary(model, input_size=(6, 64, 1024))

    for epoch in range(1, num_epochs + 1):
        train_loader, test_loader = load_dataset(root_dir=root_dir, batch_size=hyperparams['batch_size'])
        train_loss, train_t_acc, train_q_acc = train_one_epoch(
            epoch, train_loader)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Translation Acc/Train", train_t_acc, epoch)
        writer.add_scalar("Orientation Acc/Train", train_q_acc, epoch)

        test_loss, test_t_acc, test_q_acc = test_epoch(test_loader)
        writer.add_scalar("Loss/Test", test_loss, epoch)
        writer.add_scalar("Translation Acc/Test", test_t_acc, epoch)
        writer.add_scalar("Orientation Acc/Test", test_q_acc, epoch)

        if epoch % hyperparams['step_size'] == 0:
            adjust_weight_decay(optimizer, epoch)
            lr_scheduler.step()
            torch.save(model.state_dict(), f"Epoch:{epoch}, Loss: {test_loss}, Acc: {test_t_acc}{test_q_acc}_DeepPCO.pth")

    # After training, save the model
    model.to('cpu')
    model.eval()

    torch.save(model.state_dict(), "DeepPCO.pth")

    # Convert the model to torch.jit.script to load in cpp
    model_scripted = torch.jit.script(model)
    model_scripted.save("DeepPCO_scripted.pt")
    writer.close()

#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from dataloader import *
from models.DeepPCO import *

import torchsummary
from tqdm import tqdm

# define model hyperparmeters
hyperparams = {'Epoch': 50,
               'lr': 1e-4,
               'betas': [0.9, 0.999],
               'batch_size': 8,
               'wd': 1e-5,
               'step_size': 10,
               'gamma': 0.5}


def calculate_rmse(pred, gt):
    t_mse = nn.MSELoss()(pred[:, :3], gt[:, :3])
    r_mse = nn.MSELoss()(pred[:, 3:], gt[:, 3:])
    t_rmse, r_rmse = torch.sqrt(t_mse), torch.sqrt(r_mse)

    return t_rmse, r_rmse

def valid_one_epoch(valid_loader):
    model.eval()
    valid_loss = 0.0
    position_error = 0.0
    rotation_error = 0.0

    progress_bar = tqdm(valid_loader, total=len(valid_loader), desc=f'Epoch {epoch}/{num_epochs}, Valid Loss: 0.0000')
    
    with torch.no_grad():
        for batch_idx, (img, gt) in enumerate(progress_bar):
            img, gt = img.to(device), gt.to(device)
            output = model(img)
            loss = criterion(output, gt)
            valid_r_error, valid_p_error = calculate_rmse(output, gt)

            valid_loss += loss.item()
            rotation_error += valid_r_error.item()
            position_error += valid_p_error.item()

            progress_bar.set_description(f'Epoch {epoch}/{num_epochs}, Valid Loss: {valid_loss / (batch_idx + 1):.4f}, Valid position error: {valid_p_error / (batch_idx + 1):.4f}, Valid rotation error: {valid_r_error / (batch_idx + 1):.4f}')

    valid_loss /= len(valid_loader)
    position_error /= len(valid_loader)
    rotation_error /= len(valid_loader)
    progress_bar.close()

    return valid_loss, position_error, rotation_error

def train_one_epoch(epoch, train_loader):
    train_loss = 0.0
    position_error = 0.0
    rotation_error = 0.0

    model.train()
    progress_bar = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch}/{num_epochs}, Train Loss: 0.0000')

    for batch_idx, (img, gt) in enumerate(progress_bar):
        img, gt = img.to(device), gt.to(device)
        output = model(img)  # output is (roll, pitch, yaw, x, y, z)
        loss = criterion(output, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_r_error, train_p_error = calculate_rmse(output, gt)

        train_loss += loss.item()
        rotation_error += train_r_error.item()
        position_error += train_p_error.item()

        progress_bar.set_description(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss / (batch_idx + 1):.4f}, Train position error: {train_p_error / (batch_idx + 1):.4f}, Train rotation error: {train_r_error / (batch_idx + 1):.4f}')

    train_loss /= len(train_loader)
    position_error /= len(train_loader)
    rotation_error /= len(train_loader)
    progress_bar.close()

    return train_loss, position_error, rotation_error

if __name__ == '__main__':
    root_dir = '/home/smeet/catkin_ws/src/PointFlow-Odometry/dataset/custom_sequence/'
    flownt_dir = '/home/smeet/catkin_ws/src/PointFlow-Odometry/dataset/custom_sequence/flownets_EPE1.951.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepPCO().to(device)
    model.load_Flownet(flownet_dir)
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
        train_loader, valid_loader, test_loader = load_dataset(root_dir=root_dir, batch_size=hyperparams['batch_size'])
        train_loss, train_t_acc, train_q_acc = train_one_epoch(epoch, train_loader)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Translation Acc/Train", train_t_acc, epoch)
        writer.add_scalar("Orientation Acc/Train", train_q_acc, epoch)

        valid_loss, valid_t_acc, valid_q_acc = valid_one_epoch(valid_loader)
        writer.add_scalar("Loss/Test", test_loss, epoch)
        writer.add_scalar("Translation Acc/Test", test_t_acc, epoch)
        writer.add_scalar("Orientation Acc/Test", test_q_acc, epoch)

        if epoch % hyperparams['step_size'] == 0:
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

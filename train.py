import os, time, shutil, importlib
import numpy as np
import scipy.io as sio
import argparse
import torch
import torch.nn as nn
from skimage import io

from utils import AverageMeter, chw_to_hwc, hwc_to_chw
from dataset.loader import Base


parser = argparse.ArgumentParser(description = 'Train')
parser.add_argument('--model', default='gmsnet', type=str, help='model name')
parser.add_argument('--bs', default=32, type=int, help='batch size')
parser.add_argument('--ps', default=128, type=int, help='patch size')
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
parser.add_argument('--epochs', default=6000, type=int, help='sum of epochs')
parser.add_argument('--eval_freq', default=100, type=int, help='evaluation frequency')
args = parser.parse_args()


def train(train_loader1, train_loader2, model, criterion, optimizer):
	losses = AverageMeter()
	model.train()

	for (train_pairs1, train_pairs2) in zip(train_loader1, train_loader2):
		noise_img = torch.cat([train_pairs1[0], train_pairs2[0]], dim=0)
		clean_img = torch.cat([train_pairs1[1], train_pairs2[1]], dim=0)

		input_var = noise_img.cuda()
		target_var = clean_img.cuda()

		output = model(input_var)
		loss = criterion(output, target_var)
		losses.update(loss.item())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
	return losses.avg


def valid(valid_loader, model, criterion):
	losses = AverageMeter()
	model.train()

	for (noise_img, clean_img) in valid_loader:
		input_var = noise_img.cuda()
		target_var = clean_img.cuda()

		with torch.no_grad():
			output = model(input_var)

		loss = criterion(output, target_var)
		losses.update(loss.item())

	return losses.avg


if __name__ == '__main__':
	save_dir = os.path.join('./save_model/', args.model)

	model = importlib.import_module('.' + args.model, package='model').Network()
	model.cuda()
	model = nn.DataParallel(model)

	if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
		# load existing model
		model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
		print('==> loading existing model:', os.path.join(save_dir, 'checkpoint.pth.tar'))
		model.load_state_dict(model_info['state_dict'])
		optimizer = torch.optim.Adam(model.parameters())
		optimizer.load_state_dict(model_info['optimizer'])
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
		scheduler.load_state_dict(model_info['scheduler'])
		cur_epoch = model_info['epoch']
		best_loss = model_info['loss']
	else:
		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		# create model
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
		cur_epoch = 0
		best_loss = 1.0
		
	criterion = nn.L1Loss()
	criterion.cuda()

	train_dataset1 = Base('./data/SIDD_train/', 320, patch_size=args.ps)
	train_loader1 = torch.utils.data.DataLoader(
		train_dataset1, batch_size=(args.bs-args.bs//4), shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

	train_dataset2 = Base('./data/Syn_train/', 100, patch_size=args.ps)
	train_loader2 = torch.utils.data.DataLoader(
		train_dataset2, batch_size=(args.bs//4), shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

	valid_dataset = Base('./data/SIDD_valid/', 1280, cropped=False)
	valid_loader = torch.utils.data.DataLoader(
		valid_dataset, batch_size=args.bs, num_workers=8, pin_memory=True)

	for epoch in range(cur_epoch, args.epochs + 1):
		train_loss = train(train_loader1, train_loader2, model, criterion, optimizer)
		scheduler.step()

		if epoch % args.eval_freq == 0:
			avg_loss = valid(valid_loader, model, criterion)

			if avg_loss < best_loss:
				best_loss = avg_loss
				torch.save({
					'epoch': epoch + 1,
					'loss': best_loss,
					'state_dict': model.state_dict(),
					'optimizer' : optimizer.state_dict(),
					'scheduler' : scheduler.state_dict()}, 
					os.path.join(save_dir, 'best_model.pth.tar'))

		torch.save({
			'epoch': epoch + 1,
			'loss': best_loss,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict(),
			'scheduler' : scheduler.state_dict()}, 
			os.path.join(save_dir, 'checkpoint.pth.tar'))

		print('Epoch [{0}]\t'
			'lr: {lr:.6f}\t'
			'Train Loss: {train_loss:.5f}\t'
			'Best valid loss: {valid_loss:.5f}'
			.format(
			epoch,
			lr=optimizer.param_groups[-1]['lr'],
			train_loss=train_loss,
			valid_loss=best_loss))
							
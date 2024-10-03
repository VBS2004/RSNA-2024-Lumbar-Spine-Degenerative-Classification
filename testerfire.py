import argparse
import os
import random,numpy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation




def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		self.devil=nn.Sequential(
			nn.Conv2d(3, 4,3),
			nn.BatchNorm2d(4),
			nn.ReLU(),
			nn.AvgPool2d(2, 2),

			nn.Conv2d(4, 8, 3),
			nn.BatchNorm2d(8),
			nn.AvgPool2d(2, 2),

			nn.Conv2d(8, 16, 3),
			nn.BatchNorm2d(16),
			nn.AvgPool2d(2, 2),

			nn.Conv2d(16, 32, 3),
			nn.BatchNorm2d(32),
			nn.AvgPool2d(2, 2),

			nn.Conv2d(32, 64, 3),
			nn.BatchNorm2d(64),
			nn.AvgPool2d(2, 2),

			nn.Conv2d(64, 128, 3),
			nn.BatchNorm2d(128),
			nn.AvgPool2d(2, 2),

			nn.Flatten(),

			nn.Linear(512, 64),
			nn.Linear(64, 32),
			nn.Linear(32, 16),
			nn.Linear(16, 3)
			)

	def forward(self,x):
		return self.devil(x)




if __name__=="__main__":
	torch.backends.cudnn.benchmark = True
	z_w=[]
	manualSeed = 999
	print("Random Seed: ", manualSeed)
	random.seed(manualSeed)
	torch.manual_seed(manualSeed)
	torch.use_deterministic_algorithms(True)
	os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

		# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
	# in PyTorch 1.12 and later.
	torch.backends.cuda.matmul.allow_tf32 = True

	# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
	torch.backends.cudnn.allow_tf32 = True
	workers = 12
	batch_size = 128
	nz = 100
	num_epochs = 5
	lr = 0.001
	beta1 = 0.5
	ngpu=1
	ngf,nc = 3,3
	ndf = 64
	nttepoch=0
	#transforms.Resize(size=(config.INPUT_HEIGHT,config.INPUT_WIDTH))
	transform = transforms.Compose(
		[transforms.Resize(256),transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.ImageFolder(root=f"E:/rsna/neural_foraminal_narrowing",transform=transform)
	dataloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=workers)
	#testset_ = torchvision.datasets.ImageFolder(root=f"E:/rsna/neural_foraminal_narrowing",transform=transform)
	#dataloader_ = torch.utils.data.DataLoader(testset_,batch_size=,shuffle=True,num_workers=32)
	device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

	netD = Discriminator().to(device)
	if (device.type == 'cuda') and (ngpu > 1):
		netD = nn.DataParallel(netD, list(range(ngpu)))
	netD.apply(weights_init)
	try:
		netD.load_state_dict(torch.load("E:/rsna/best-model.pt"))
		print("Loaded")
	except:
		pass
	#netD.load_state_dict(torch.load("E:/rsna/best-model.pt", map_location=device))

	criterion,img_devil = nn.CrossEntropyLoss(),0
	fixed_noise = torch.randn(1, nz, 1, 1, device=device)

	optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
	schedulerD=torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.86)
	#schedulerD=torch.optim.lr_scheduler.OneCycleLR(optimizerD)
	print(dataloader.dataset.classes)


	img_list = []
	G_losses = []
	D_losses = []
	iters,z_ = 0,0
	lr_i=0
	print("Starting Training Loop...")
	best_performance = 0  # Variable to track the best performance so far
	model_path = "E:/rsna/best-modelMax.pt" # Path to save the best model
	least_loss=float('inf')
	while(True):
		epoch_loss = 0.0
		z_, z,class_0c,class_1c,class2_c,class_0w,class_1w,class2_w = 0, 0,0,0,0,0,0,0

		for i, data in enumerate(dataloader, 0):
			nttepoch+=1
			optimizerD.zero_grad() #Added line
			real_cpu = data[0].to(device)
			label = data[1].to(device)
			output = netD(real_cpu)
			errD_real = criterion(output, label)
			epoch_loss+=errD_real.item()
			label_=label
			output = torch.argmax(netD(real_cpu),dim=1).view(-1)
			correct_arr=(label.detach().cpu().numpy().reshape(-1) == output.detach().cpu().numpy())
			z+=correct_arr.sum()
			for i,j in zip(output.detach().cpu().numpy(),correct_arr):
				if j:
					if i==0:
						class_0c+=1
					elif i==1:
						class_1c+=1
					elif i==2:
						class2_c+=1
				else:
					if i==0:
						class_0w+=1
					elif i==1:
						class_1w+=1
					elif i==2:
						class2_w+=1
			z_ += len(label)
			if(nttepoch==1):
				print("Label:",label)
				print("Output:",output)
				print("label.detach().cpu().numpy().reshape(-1) == output.detach().cpu().numpy():")
				print(label.detach().cpu().numpy().reshape(-1))
				print(output.detach().cpu().numpy())
		print(f"Loss:{epoch_loss/len(dataloader)} \nClass 0: {class_0c}/{class_0w+class_0c} Class 1: {class_1c}/{class_1c+class_1w} Class 2: {class2_c}/{class2_w+class2_c}")
		
		print("Sample went throught:",z_,"Classified correctly:",z)
		z_w.append(z)
		if len(z_w)>=4:
			if len([True for i in range(1,3) if z_w[len(z_w)-i]<=z_w[len(z_w)-3]+2 and z_w[len(z_w)-i]>=z_w[len(z_w)-4]-3])==2:
				print([True for i in range(1,3) if z_w[len(z_w)-i]<=z_w[len(z_w)-3]+2 and z_w[len(z_w)-i]>=z_w[len(z_w)-4]-3])
				z_w=[]
				print(optimizerD.param_groups[0]["lr"])
				schedulerD.step()
				print(optimizerD.param_groups[0]["lr"])


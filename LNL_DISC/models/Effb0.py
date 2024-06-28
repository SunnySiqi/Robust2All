import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class Effb0(nn.Module):
	def __init__(self, num_classes=1):
		super().__init__()
		self.num_classes = num_classes
		net = models.efficientnet_b0(weights='DEFAULT')
		layers = list(net.children())
		layers[0][0][0] = nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1, bias=False)

		layers = list(net.children())[:-1]
		self.net = nn.Sequential(*layers)
		self.fc = nn.Linear(1280, num_classes)
		self.proxies = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(num_classes, 1280)))

		self.params = nn.ParameterDict({
			'base': list(self.net.parameters()),
			'classifier': self.proxies})

	def forward(self, x):
		z = self.net(x)
		feature = z.view(z.size(0), -1)
		y_hat = self.fc(feature)
		#x_hat = F.linear(feature, self.proxies)
		return y_hat
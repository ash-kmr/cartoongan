import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
class Residual_block(nn.Module):
	def __init__(self, in_channels, k_size, stride, padding):
		super(Residual_block, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, in_channels, k_size, stride, padding)
		self.norm1 = nn.InstanceNorm2d(in_channels)
		self.conv2 = nn.Conv2d(in_channels, in_channels, k_size, stride, padding)
		self.norm2 = nn.InstanceNorm2d(in_channels)

	def forward(self, inp):
		out = self.norm1(self.conv1(inp))
		out = F.relu(x)
		out = self.norm2(self.conv2(inp))
		return inp+out

class Generator(nn.Module):
	"""
	out_channels: these are the output channels for the very first convolutional
	layer in paper.
	"""
	def __init__(self, in_channels, out_channels, k_size, stride, padding):
		super(Generator, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, k_size, stride, 3*padding)
		self.norm1 = nn.InstanceNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, 2*out_channels, k_size//2, 2*stride, padding)
		self.conv3 = nn.Conv2d(2*out_channels, 2*out_channels, k_size//2, stride, padding)
		self.norm2 = nn.InstanceNorm2d(2*out_channels)
		self.conv4 = nn.Conv2d(2*out_channels, 4*out_channels, k_size//2, 2*stride, padding)
		self.conv5 = nn.Conv2d(4*out_channels, 4*out_channels, k_size//2, stride, padding)
		self.norm3 = nn.InstanceNorm2d(4*out_channels)
		self.residual_blocks = [Residual_block(256, 3, 1, padding) for i in range(8)]
		self.upconv1 = nn.ConvTranspose2d(4*out_channels, 2*out_channels, k_size//2, 2*stride, padding)
		self.upconv2 = nn.Conv2d(2*out_channels, 2*out_channels, k_size//2, stride, padding)
		self.upnorm1 = nn.InstanceNorm2d(2*out_channels)
		self.upconv3 = nn.ConvTranspose2d(2*out_channels, out_channels, k_size//2, 2*stride, padding)
		self.upconv4 = nn.Conv2d(out_channels, out_channels, k_size//2, stride, padding)
		self.upnorm2 = nn.InstanceNorm2d(out_channels)
		self.upconv5 = nn.Conv2d(out_channels, in_channels, k_size, stride, 3*padding)


	def forward(self, x):
		x = nn.ReLU(self.norm1(self.conv1(x)))
		x = nn.ReLU(self.norm2(self.conv3(self.conv2(x))))
		x = nn.ReLU(self.norm3(self.conv5(self.conv4(x))))
		for resnet in residual_blocks:
			x = resnet(x)
		x = nn.ReLU(self.upnorm1(self.upconv2(self.upconv1(x))))
		x = nn.ReLU(self.upnorm2(self.upconv4(self.upconv3(x))))
		x = nn.ReLU(self.upconv5(x))

		return x


gen = Generator(3, 64, 7, 1, 1)

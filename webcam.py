from threading import Thread
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from PIL import Image
import torchvision.transforms.functional as TF
import cv2


class Webcam:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
    	# start the thread to read frames from the video stream
    	Thread(target=self.update, args=()).start()
    	return self

    def update(self):
    	# keep looping infinitely until the thread is stopped
    	while True:
    		# if the thread indicator variable is set, stop the thread
    		if self.stopped:
    			return

    		# otherwise, read the next frame from the stream
    		(self.grabbed, self.frame) = self.stream.read()

    def read(self):
    	# return the frame most recently read
    	return self.frame

    def stop(self):
    	# indicate that the thread should be stopped
    	self.stopped = True

class SSIMLoss(nn.Module):
    def __init__(self, device, window_size=11, size_average=True, channels=3):
        super(SSIMLoss, self).__init__()
        self.device = device
        self.window_size = window_size
        self.size_average = size_average
        self.window = self._create_window(window_size, channels).to(self.device)
        self.channels = channels

    def _create_window(self, window_size, channels):
        gaussian = torch.Tensor([exp(-(x - window_size//2)**2/float(2*1.5**2)) for x in range(window_size)])
        gaussian = (gaussian/gaussian.sum()).unsqueeze(1)
        _2D_window = gaussian.mm(gaussian.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channels, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2):
        mu_x = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channels)
        mu_y = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channels)

        mu_x2 = mu_x**2
        mu_y2 = mu_y**2
        mu_xy =  mu_x*mu_y

        sigma_x2 = F.conv2d(img1*img1, self.window, padding=self.window_size//2, groups=self.channels) - mu_x2
        sigma_y2 = F.conv2d(img2*img2, self.window, padding=self.window_size//2, groups=self.channels) - mu_y2
        sigma_xy = F.conv2d(img1*img2, self.window, padding=self.window_size//2, groups=self.channels) - mu_xy

        K1 = 0.01
        K2 = 0.03

        L = 1.0

        C1 = (K1*L)**2
        C2 = (K2*L)**2

        ssim_map = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2))/((mu_x2 + mu_y2 + C1)*(sigma_x2 + sigma_y2 + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        img1 = img1.permute(2, 1, 0).unsqueeze(0)
        img2 = img2.permute(2, 1, 0).unsqueeze(0).detach()

        return -self._ssim(img1, img2)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        target_feature = target_feature.permute(2, 1, 0).unsqueeze(0)
        self.target = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        print(a,b,c,d)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.reshape(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def forward(self, input):
        input = input.permute(2, 1, 0).unsqueeze(0)
        G = self.gram_matrix(input)
        loss = F.mse_loss(G, self.target)
        return loss

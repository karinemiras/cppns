import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
import numpy as np
import pyaudio
import cv2
import time
from torchvision.transforms import ToTensor
from PIL import Image
import copy
import glob
from threading import Thread
from webcam import Webcam, SSIMLoss, StyleLoss

class Network(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, non_linearity, uniform=5.0):
        super(Network, self).__init__()

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.non_linearity = non_linearity

        self.module_list = nn.ModuleList()
        self.module_list.append(nn.Linear(self.n_inputs, self.n_hidden[0]))

        previous = self.n_hidden[0]

        for hidden_layer_size in self.n_hidden[1:]:
            lin = nn.Linear(previous, hidden_layer_size)
            nn.init.normal_(lin.weight, std=uniform)
            nn.init.normal_(lin.bias, std=uniform)
            self.module_list.append(lin)
            previous = hidden_layer_size

        self.module_list.append(nn.Linear(previous, self.n_outputs))

    def forward(self, x):
        for i in range(len(self.module_list) - 1):
            x = self.non_linearity(self.module_list[i](x))
        return torch.sigmoid(self.module_list[-1](x))

def initialise_device():
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    print('Using %s' % device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
    return device

class Sound:
    def __init__(self, CHUNK=2048, FORMAT=pyaudio.paInt16, CHANNELS=1, RATE=44100):
        self.CHUNK = CHUNK

        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        self.previous_bands = np.random.rand(8)
        self.gain = 0.01
        self.alpha = 0.8

        self.sin = False
        self.amplitudes = np.abs(np.random.rand(8))
        self.frequencies = np.abs(np.random.rand(8))

        self.calculate_bands = False

    def read_audio(self):
        while True:
            data = self.stream.read(self.CHUNK)

            decoded = np.fromstring(data, dtype=np.int16)

            frequencies = np.fft.fft(decoded)

            current_bins = self.bands(np.abs(frequencies[:int(self.CHUNK/2)]), self.sin, self.amplitudes, self.frequencies)

            yield current_bins

    def bands(self, amplitudes, sin, sin_amplitudes, sin_frequencies):
        if self.calculate_bands:
            bands = np.zeros(8, dtype=np.float32)
            bands[0] = np.sum(amplitudes[0:4]) * 0.1
            bands[1] = np.sum(amplitudes[4:12]) * 0.2
            bands[2] = np.sum(amplitudes[12:28]) * 0.3
            bands[3] = np.sum(amplitudes[28:60])
            bands[4] = np.sum(amplitudes[60:124])
            bands[5] = np.sum(amplitudes[124:252])
            bands[6] = np.sum(amplitudes[252:508])
            bands[7] = np.sum(amplitudes[508:])

            median_bands = np.median(bands)

            normalized_bands = self.gain*bands/(median_bands + np.finfo(float).eps)

            if sin:
                normalized_bands += sin_amplitudes*np.sin(0.5*sin_frequencies*[time.time()])

            smoothed_bands = self.alpha*self.previous_bands + (1-self.alpha)*normalized_bands

            self.previous_bands = smoothed_bands

            return smoothed_bands
        else:
            return np.ones(8) + sin_amplitudes*np.sin(0.5*sin_frequencies*[time.time()])


class CPPN:
    def __init__(self, height, width, n_inputs, n_hidden, n_outputs, non_linearity, device, webcam=False):
        self.height = height
        self.width = width

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.non_linearity = non_linearity
        self.device = device

        self.network = Network(n_inputs, n_hidden, n_outputs, non_linearity)
        self.network.to(device)

        self.network_new = Network(n_inputs, n_hidden, n_outputs, non_linearity)
        self.network_new.to(device)

        self.sound = Sound()

        self.visualisation_input = self._create_visualisation_tensor()

        if webcam:
            self.webcam = Webcam()
            self.webcam.start()

    def _create_visualisation_tensor(self):
        visualisation_input = np.zeros((self.height, self.width, self.n_inputs))

        for i in range(self.height):
            for j in range(self.width):
                visualisation_input[i, j] = [i/float(self.height),j/float(self.width)] + [0]*(self.n_inputs-2)

        visualisation_input = torch.tensor(visualisation_input.reshape(-1, self.n_inputs), dtype=torch.float, device=self.device)

        return visualisation_input

    # def _create_kaleidoscope_visualisation_tensor(self):
    #     visualisation_input = np.zeros((self.height, self.width, self.n_inputs))
    #
    #     for i in range(self.height):
    #         for j in range(self.width):
    #             visualisation_input[i, j] = [i/float(self.height),j/float(self.width)] + [0]*(self.n_inputs-2)
    #
    #     visualisation_input = torch.tensor(visualisation_input.reshape(-1, self.n_inputs), dtype=torch.float, device=self.device)
    #
    #     return visualisation_input

    def _visualise_np(self, bands=None):
        with torch.no_grad():
            if bands is not None:
                o = torch.tensor(bands, dtype=torch.float).repeat(self.visualisation_input.size(0),1)

                self.visualisation_input[:,2:] = o

            im = self.network(self.visualisation_input).reshape(self.width, self.height, self.n_outputs)

            return im.detach().cpu().numpy()

    def _visualise(self, name, save_im=False):
        with torch.no_grad():
            im = self.network(self.visualisation_input).reshape(self.width, self.height, 3).detach().cpu().numpy()*255

            if save_im:
                cv2.imwrite(name + ".png", im)

            return im

    def start(self):
        cv2.namedWindow("CPPN", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("CPPN", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        audio_generator = self.sound.read_audio()

        t = time.time()
        interpolating = 0
        training = 0
        generator = None

        while True:
            if time.time()-t > 20. and not training and not interpolating:
                choice = np.random.choice([1])

                if choice==0:
                    training = 1
                    target_images = glob.glob("target_images/*")
                    choice = np.random.choice(target_images)
                    image = cv2.imread(choice)
                    image = ToTensor()(image).float().to(self.device)

                    generator = self.train(image, self.height, self.width, 0.001, save_im=False, image_folder="formation1", network_name="Santa",epochs=100, live=True, verbose=True)
                elif choice==1:
                    self._random_network()
                    generator = self.interpolate(60)
                    interpolating = 1
                elif choice==2:
                    print("webcam")
                    training = 1
                    image = self.webcam.read()

                    image = ToTensor()(image).float().to(self.device)

                    generator = self.train(image, self.height, self.width, 0.001, save_im=False, image_folder="formation1", network_name="Santa",epochs=100, live=True, verbose=True)

                if choice==0:
                    self.sound.sin = self.sound.sin==False
                    self.sound.amplitudes = np.abs(np.random.rand(8))
                    self.sound.frequencies = np.abs(np.random.rand(8))
                elif choice==1:
                    self.sound.calculate_bands = self.sound.calculate_bands==False

                if self.sound.calculate_bands==False and self.sound.sin==False:
                    if np.random.rand()<0.5:
                        self.sound.calculate_bands = True
                    else:
                        self.sound.sin = True

                t = time.time()

            if interpolating:
                try:
                    self.network = next(generator)
                except StopIteration:
                    interpolating = 0

            if training:
                try:
                    self.network = next(generator)
                except StopIteration:
                    training = 0
                    torch.cuda.empty_cache()


            frame = self._visualise_np(next(audio_generator))
            cv2.imshow("CPPN", cv2.copyMakeBorder(frame, 0, 0, 0, 0, cv2.BORDER_CONSTANT, 0))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.sound.stream.stop_stream()
        self.sound.stream.close()
        self.sound.pa.terminate()

    def _resize_image(self, image, width, height):
        image = F.interpolate(image.unsqueeze(0), size=(width, height))
        image = torch.squeeze(image, 0)
        image = image.permute(1,2,0)

        return image

    def _random_network(self):
        self.network_new = Network(self.n_inputs, self.n_hidden, self.n_outputs, self.non_linearity)
        self.network_new.to(self.device)

    def train(self, image, width, height, loss_threshold, save_im=False, image_folder="", save_network=True, network_name="", epochs=1, live=False, verbose=False):
        image = self._resize_image(image, width, height)

        optimiser = torch.optim.Adam(self.network.parameters())
        criterion = torch.nn.MSELoss()

        previous_loss = None

        network = copy.copy(self.network)

        for i in range(epochs):
            loss = criterion(network(self.visualisation_input).reshape(width, height, 3), image)

            optimiser.zero_grad()
            loss.backward()

            optimiser.step()
            if verbose:
                print(i, loss)
            if save_im:
                if previous_loss is None or previous_loss - loss_threshold > loss:
                    previous_loss = loss
                    self._visualise("{}/image{:06d}.png".format(image_folder, i), save_im)
            if live:
                yield network

        if save_network:
            torch.save(self.network.state_dict(), 'trained_networks/network{}.pt'.format(network_name))

        torch.cuda.empty_cache()

    def load(self, network_name, new=False):
        if new:
            self.network_new.load_state_dict(torch.load("trained_networks/network" + network_name + ".pt"))
            self.network_new.to(self.device)
        else:
            self.network.load_state_dict(torch.load("trained_networks/network" + network_name + ".pt"))
            self.network.to(self.device)

    def interpolate(self, num_interpolation_frames, space="beta", beta=2.0):
        """
        beta parameter alters the steepness of the function
        """
        weights1 = []
        biases1 = []

        weights2 = []
        biases2 = []

        for idx, layer in enumerate(self.network.module_list):
            weights1.append(layer.weight.data)
            biases1.append(layer.bias.data)
            weights2.append(self.network_new.module_list[idx].weight.data)
            biases2.append(self.network_new.module_list[idx].bias.data)

        frame_distribution = frame_distribution = np.linspace(0, 1, num=num_interpolation_frames)

        if space!="lin":
            f = lambda x: 1/(1+np.power(x/(1+np.finfo(float).eps-x),-beta))
            frame_distribution = f(frame_distribution)

        weights = []
        for weight_pair in zip(weights1, weights2):
            difference = weight_pair[0] - weight_pair[1]
            new_weights = torch.zeros(num_interpolation_frames, list(weight_pair[0].size())[0], list(weight_pair[0].size())[1]).float().to(self.device)
            for idx, i in enumerate(frame_distribution):
                new_weights[idx] = weight_pair[0] - i*difference
            weights.append(new_weights)

        biases = []
        for bias_pair in zip(biases1, biases2):
            difference = bias_pair[0] - bias_pair[1]
            new_biases = torch.zeros(num_interpolation_frames, list(bias_pair[0].size())[0]).float().to(self.device)
            for idx, i in enumerate(frame_distribution):
                new_biases[idx] = bias_pair[0] - i*difference
            biases.append(new_biases)

        network = copy.deepcopy(self.network)

        for i in range(num_interpolation_frames):
            for idx, layer in enumerate(network.module_list):
                network.module_list[idx].weight.data = weights[idx][i]
                network.module_list[idx].bias.data = biases[idx][i]

            yield network

    def random_walk(self):
        pass

    def animate(self):
        pass

    def alzheimer(self):
        pass

def main():
    # Network parameters
    n_inputs = 10
    n_hidden = [100]*4
    n_outputs = 3

    non_linearity = torch.tanh

    device = initialise_device()

    # Image parameters
    height = 512
    width = 512

    webcam = True

    cppn = CPPN(height, width, n_inputs, n_hidden, n_outputs,non_linearity, device, webcam)
    cppn.start()

if __name__=="__main__":
    main()

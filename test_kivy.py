from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle
from kivy.clock import Clock
from kivy.core.window import Window
from scipy.io.wavfile import write
import wave
import numpy as np
import time, math, scipy
from collections import deque
from scipy.signal import savgol_filter

from src.fft import getFFT
from src.utils import *

from matplotlib import cm

class AudioVisualizerWidget(Widget,):
    def __init__(self, 
                device = None,
                rate   = None,
                FFT_window_size_ms  = 50,
                updates_per_second  = 100,
                smoothing_length_ms = 50,
                n_frequency_bins    = 51,
                visualize = True,
                verbose   = False,
                height    = 450,
                window_ratio = 24/9
                ,**kwargs):
        super().__init__(**kwargs)
        self.audio_data = []
        self.n_frequency_bins = n_frequency_bins
        self.rate = rate
        self.verbose = verbose
        self.visualize = visualize
        self.height = height
        self.window_ratio = window_ratio

        try:
            from src.stream_reader_pyaudio import Stream_Reader
            self.stream_reader = Stream_Reader(
                device  = device,
                rate    = rate,
                updates_per_second  = updates_per_second,
                verbose = verbose)
        except:
            from src.stream_reader_sounddevice import Stream_Reader
            self.stream_reader = Stream_Reader(
                device  = device,
                rate    = rate,
                updates_per_second  = updates_per_second,
                verbose = verbose)
            print("ok2")
        self.rate = self.stream_reader.rate
        #Custom settings:
        self.rolling_stats_window_s    = 20     # The axis range of the FFT features will adapt dynamically using a window of N seconds
        self.equalizer_strength        = 0.20   # [0-1] --> gradually rescales all FFT features to have the same mean
        self.apply_frequency_smoothing = True   # Apply a postprocessing smoothing filter over the FFT outputs

        if self.apply_frequency_smoothing:
            self.filter_width = round_up_to_even(0.03*self.n_frequency_bins) - 1

        self.FFT_window_size = round_up_to_even(self.rate * FFT_window_size_ms / 1000)
        self.FFT_window_size_ms = 1000 * self.FFT_window_size / self.rate
        self.fft  = np.ones(int(self.FFT_window_size/2), dtype=float)
        self.fftx = np.arange(int(self.FFT_window_size/2), dtype=float) * self.rate / self.FFT_window_size

        self.data_windows_to_buffer = math.ceil(self.FFT_window_size / self.stream_reader.update_window_n_frames)
        self.data_windows_to_buffer = max(1,self.data_windows_to_buffer)

        # Temporal smoothing:
        # Currently the buffer acts on the FFT_features (which are computed only occasionally eg 30 fps)
        # This is bad since the smoothing depends on how often the .get_audio_features() method is called...
        self.smoothing_length_ms = smoothing_length_ms
        if self.smoothing_length_ms > 0:
            self.smoothing_kernel = get_smoothing_filter(self.FFT_window_size_ms, self.smoothing_length_ms, verbose=1)
            self.feature_buffer = numpy_data_buffer(len(self.smoothing_kernel), len(self.fft), dtype = np.float32, data_dimensions = 2)

        #This can probably be done more elegantly...
        self.fftx_bin_indices = np.logspace(np.log2(len(self.fftx)), 0, len(self.fftx), endpoint=True, base=2, dtype=None) - 1
        self.fftx_bin_indices = np.round(((self.fftx_bin_indices - np.max(self.fftx_bin_indices))*-1) / (len(self.fftx) / self.n_frequency_bins),0).astype(int)
        self.fftx_bin_indices = np.minimum(np.arange(len(self.fftx_bin_indices)), self.fftx_bin_indices - np.min(self.fftx_bin_indices))

        self.frequency_bin_energies = np.zeros(self.n_frequency_bins)
        self.frequency_bin_centres  = np.zeros(self.n_frequency_bins)
        self.fftx_indices_per_bin   = []
        for bin_index in range(self.n_frequency_bins):
            bin_frequency_indices = np.where(self.fftx_bin_indices == bin_index)
            self.fftx_indices_per_bin.append(bin_frequency_indices)
            fftx_frequencies_this_bin = self.fftx[bin_frequency_indices]
            self.frequency_bin_centres[bin_index] = np.mean(fftx_frequencies_this_bin)

        #Hardcoded parameters:
        self.fft_fps = 30
        self.log_features = False   # Plot log(FFT features) instead of FFT features --> usually pretty bad
        self.delays = deque(maxlen=20)
        self.num_ffts = 0
        self.strongest_frequency = 0

        #Assume the incoming sound follows a pink noise spectrum:
        self.power_normalization_coefficients = np.logspace(np.log2(1), np.log2(np.log2(self.rate/2)), len(self.fftx), endpoint=True, base=2, dtype=None)
        self.rolling_stats_window_n = self.rolling_stats_window_s * self.fft_fps #Assumes ~30 FFT features per second
        self.rolling_bin_values = numpy_data_buffer(self.rolling_stats_window_n, self.n_frequency_bins, start_value = 25000)
        self.bin_mean_values = np.ones(self.n_frequency_bins)

        print("Using FFT_window_size length of %d for FFT ---> window_size = %dms" %(self.FFT_window_size, self.FFT_window_size_ms))
        print("##################################################################################################")

        #Let's get started:
        self.stream_reader.stream_start(self.data_windows_to_buffer)

        if self.visualize:
            self.HEIGHT = round(self.height)
            self.WIDTH  = round(self.window_ratio*self.HEIGHT)
            self.y_ext = [round(0.05*self.HEIGHT), self.HEIGHT]
            self.cm = cm.plasma
            
            self.decay_speed        = 0.06
            self.inter_bar_distance = int(0.2*self.WIDTH / self.n_frequency_bins)
            self.avg_energy_height  = 0.225
            self.bar_width = (self.WIDTH / self.n_frequency_bins) - self.inter_bar_distance
            #Configure the bars:
            self.slow_bars, self.fast_bars, self.bar_x_positions = [],[],[]
            for i in range(self.n_frequency_bins):
                x = int(i* self.WIDTH / self.n_frequency_bins)
                
                fast_bar = [int(x), int(self.y_ext[0]), math.ceil(self.bar_width), 10]
                slow_bar = [int(x), None, math.ceil(self.bar_width), 10]
                #print(fast_bar)
                self.bar_x_positions.append(x)
                self.fast_bars.append(fast_bar)
                self.slow_bars.append(slow_bar)
                self.slow_bar_thickness = max(0.00002*self.HEIGHT, 1.25 / self.n_frequency_bins)
                self.tag_every_n_bins = max(1,round(5 * (self.n_frequency_bins / 51)))

            self.slow_features = [0]*self.n_frequency_bins
            self.frequency_bin_max_energies  = np.zeros(self.n_frequency_bins)
            self.frequency_bin_energies = self.frequency_bin_energies
            self.bin_text_tags, self.bin_rectangles = [], []

            #Fixed init params:
            self.start_time = None
            self.vis_steps  = 0
            self.fps_interval = 10
            self.fps = 0
            self._is_visualizing = False
        
        Window.size = (self.WIDTH, self.HEIGHT)
        self.bar = []
        # pos = [(0, 22),(12, 22),(24, 22)]
        # size = [(10, 50),(10, 60),(10, 70)]
        self.init_bars()

    def init_bars(self):
        with self.canvas:
            for bar in self.fast_bars:
                rec = Rectangle(pos=(bar[0], bar[1]), size=(bar[2], bar[3]))
                self.bar.append(rec)
    
    def start_visual(self):
        self._is_visualizing = True
    def stop_visual(self):
        self._is_visualizing = False

    def update_visual(self):
        if np.min(self.bin_mean_values) > 0:
            self.frequency_bin_energies = self.avg_energy_height * self.frequency_bin_energies / self.bin_mean_values
        if self.start_time is None:
           self.start_time = time.time()

        self.vis_steps += 1

        if self.vis_steps%self.fps_interval == 0:
            self.fps = self.fps_interval / (time.time()-self.start_time)
            self.start_time = time.time()
        self.plot_bars_visual()
    
    def plot_bars_visual(self):
        new_slow_features = []
        local_height = self.y_ext[1] - self.y_ext[0]
        feature_values = self.frequency_bin_energies[::-1]
        #print(feature_values)
        for i in range(len(self.frequency_bin_energies)):
            #fast bar visual update
            feature_value = feature_values[i] * local_height
            #slow bar visual update
            self.fast_bars[i][3] = int(feature_value)
            self.decay = min(0.99, 1 - max(0,self.decay_speed * 60 / self.fft_fps))
            slow_feature_value = max(self.slow_features[i]*self.decay, feature_value)
            new_slow_features.append(slow_feature_value)
            #self.slow_bars[i][1] = int(self.fast_bars[i][1] + slow_feature_value)
            #self.slow_bars[i][3] = int(self.slow_bar_thickness * local_height)
            self.slow_bars[i][3] = int(slow_feature_value)
        for i, fast_bar in enumerate(self.fast_bars):
            #print(i, fast_bar)
            cur_w = self.bar[i].size[0]
            cur_h = fast_bar[3] * 0.3
            self.bar[i].size = (cur_w, cur_h)
        ##Visualizes slow bars
        # for i, slow_bar in enumerate(self.slow_bars):
        #     #print(i, slow_bar)
        #     cur_w = self.bar[i].size[0]
        #     cur_h = slow_bar[3] * 0.3
        #     self.bar[i].size = (cur_w, cur_h)

        self.slow_features = new_slow_features
    
    def update_rolling_stats(self):
        self.rolling_bin_values.append_data(self.frequency_bin_energies)
        self.bin_mean_values  = np.mean(self.rolling_bin_values.get_buffer_data(), axis=0)
        self.bin_mean_values  = np.maximum((1-self.equalizer_strength)*np.mean(self.bin_mean_values), self.bin_mean_values)

    def update_features(self, n_bins = 3):

        latest_data_window = self.stream_reader.data_buffer.get_most_recent(self.FFT_window_size)
        self.audio_data.append(latest_data_window)
        self.fft = getFFT(latest_data_window, self.rate, self.FFT_window_size, log_scale = self.log_features)
        #Equalize pink noise spectrum falloff:
        self.fft = self.fft * self.power_normalization_coefficients
        self.num_ffts += 1
        self.fft_fps  = self.num_ffts / (time.time() - self.stream_reader.stream_start_time)

        if self.smoothing_length_ms > 0:
            self.feature_buffer.append_data(self.fft)
            buffered_features = self.feature_buffer.get_most_recent(len(self.smoothing_kernel))
            if len(buffered_features) == len(self.smoothing_kernel):
                buffered_features = self.smoothing_kernel * buffered_features
                self.fft = np.mean(buffered_features, axis=0)

        self.strongest_frequency = self.fftx[np.argmax(self.fft)]

        #ToDo: replace this for-loop with pure numpy code
        for bin_index in range(self.n_frequency_bins):
            self.frequency_bin_energies[bin_index] = np.mean(self.fft[self.fftx_indices_per_bin[bin_index]])

        #Beat detection ToDo:
        #https://www.parallelcube.com/2018/03/30/beat-detection-algorithm/
        #https://github.com/shunfu/python-beat-detector
        #https://pypi.org/project/vamp/

        return

    def get_audio_features(self, dt):

        if self.stream_reader.new_data:  #Check if the stream_reader has new audio data we need to process
            self.update_features()
            self.update_rolling_stats()
            self.stream_reader.new_data = False
            self.frequency_bin_energies = np.nan_to_num(self.frequency_bin_energies, copy=True)
            if self.apply_frequency_smoothing:
                if self.filter_width > 3:
                    self.frequency_bin_energies = savgol_filter(self.frequency_bin_energies, self.filter_width, 3)
            self.frequency_bin_energies[self.frequency_bin_energies < 0] = 0
            if self.visualize:
                self.update_visual()
        return
    

class MyApp(App):
    v = AudioVisualizerWidget()
    def build(self):
        Window.bind(on_request_close=self.on_request_close)
        Clock.schedule_interval(self.v.get_audio_features, 0)
        return self.v
    def on_request_close(self, *args):
        self.v.quit()
        return True
if __name__ == "__main__":
    app = MyApp()
    app.run()
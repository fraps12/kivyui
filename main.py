import fnmatch
import os
import numpy as np
import argparse
import cv2.cv2 as cv2
import datetime
import glob
import threading, queue


from sklearn.cluster import KMeans
from skimage import exposure
from skimage import io

from glob import glob
from kivy.config import Config
from time import time
from kivy.uix.widget import Widget
from kivy.app import App
from os.path import dirname, join
from kivy.lang import Builder
from kivy.properties import NumericProperty, StringProperty, BooleanProperty, \
    ListProperty
from kivy.logger import Logger
from kivy.clock import Clock, _default_time as time, mainthread
from kivy.animation import Animation

from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics.texture import Texture
from kivy.uix.image import AsyncImage
from kivy.uix.image import Image
from kivy.uix.progressbar import ProgressBar
from kivy.uix.label import Label
from threading import Thread
from kivy.uix.checkbox import CheckBox
from kivy.clock import mainthread
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button


Window.size = (1920, 1080)
Window.fullscreen = 'auto'
Config.set('kivy','window_icon','data/kiwi.ico')

# center_x: self.parent.center_x


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)


# class ImageLoad(FloatLayout):

# img = StringProperty(None)
class CustomDropDown(DropDown):
    pass


class ShowcaseScreen(Screen):
    fullscreen = BooleanProperty(True)
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty()
    var_len = NumericProperty(1)
    path_text = StringProperty()
    path_text_load = StringProperty()
    path_onscreen_line = StringProperty()
    draw_line = ObjectProperty()
    consommables = ListProperty()
    wimg = ObjectProperty()
    IMG_GLOB = StringProperty('dir')
    status_icon = ObjectProperty()
    counter_ticks = NumericProperty()
    counter_files_indir = StringProperty()
    default_image = StringProperty('dir')
    default_image_result = StringProperty('dir')
    image_preview = StringProperty()
    image_preview_result = StringProperty()
    bgr_is_active = BooleanProperty(False)
    hsv_is_active = BooleanProperty(False)
    lab_is_active = BooleanProperty(False)
    YCrCb_is_active = BooleanProperty(False)
    colorspace_arg = StringProperty('BGR')

    blue_is_active = BooleanProperty(False)
    green_is_active = BooleanProperty(False)
    red_is_active = BooleanProperty(False)
    channel_arg = StringProperty()

    def bgr_activate(self):
        self.bgr_is_active = True
        self.hsv_is_active = False
        self.YCrCb_is_active = False
        self.lab_is_active = False

    def hsv_activate(self):
        self.bgr_is_active = False
        self.hsv_is_active = True
        self.YCrCb_is_active = False
        self.lab_is_active = False

    def lab_activate(self):
        self.bgr_is_active = False
        self.hsv_is_active = False
        self.YCrCb_is_active = False
        self.lab_is_active = True

    def YCrCb_activate(self):
        self.bgr_is_active = False
        self.hsv_is_active = False
        self.YCrCb_is_active = True
        self.lab_is_active = False

    def check_colorspace(self):
        if self.bgr_is_active == True:
            self.colorspace_arg = 'BGR'
        if self.hsv_is_active == True:
            self.colorspace_arg = 'HSV'
        if self.lab_is_active == True:
            self.colorspace_arg = 'Lab'
        if self.YCrCb_is_active == True:
            self.colorspace_arg = 'YCrCb'

    def activate_blue(self, value):
        self.blue_is_active = value

    def activate_green(self, value):
        self.green_is_active = value

    def activate_red(self, value):
        self.red_is_active = value

    def check_channels(self):
        if self.blue_is_active:
            self.channel_arg += '0'
        if self.green_is_active:
            self.channel_arg += '1'
        if self.red_is_active:
            self.channel_arg += '2'
        if not any([self.red_is_active, self.green_is_active, self.blue_is_active]):
            self.channel_arg = '012'

    def dismiss_channels(self):
        self.channel_arg = ''

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_save(self):
        content = SaveDialog(save=self.save, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def thr_im_widget(self):
        im_script = Thread(target=self.image_widget)
        im_script.daemon = True
        im_script.start()

    def delete_widget(self):
        self.IMG_GLOB = "dir"
        self.status_icon = Image(source=self.IMG_GLOB, pos_hint={'top': .94, 'right': .08},
                                 size_hint=(None, None), size=(32, 32), id='test')


    def image_widget(self):
        self.IMG_GLOB = 'dir'
        # , pos_hint={'top': .94, 'right': .08},
        # size_hint=(None,None), size=(32,32), id='test')
    def image_get(self):
        self.image_preview = Image(source = self.default_image, pos = ('400dp', '300dp'))
        self.add_widget(self.image_preview)
        self.image_preview_result = Image(source = self.default_image_result, pos = ('400dp', '-200dp'))
        self.add_widget(self.image_preview_result)


    def read_dirs(self, path):

        global configfiles
        #try:
        configfiles = [os.path.join(dirpath, f)
                          for dirpath, dirnames, files in os.walk(path)
                          for f in fnmatch.filter(files, '*.jpg')]
        #except Error
        self.var_len = len(configfiles)
        print(self.var_len)
        dropdown = CustomDropDown()
        mainbutton = Button(text='Hello', size_hint=(None, None))
        mainbutton.bind(on_release=dropdown.open)
        dropdown.bind(on_select=lambda instance, x: setattr(mainbutton, 'text', x))



    def red_mark(self):
        self.wimg = "dir"

    def button_script(self, f_path, c_space, ch, cl, l_path):

        thr_script = Thread(target=self.start_script, args=(f_path, c_space, ch, cl, l_path))
        thr_script.daemon = True
        thr_script.start()

    # script for image K-mean processing
    def start_script(self, first_path, color_space, channels_arg, clusters, last_path):
        print(self.channel_arg)

        for x in range(0, self.var_len):

            self.counter_ticks = x + 1
            path_name = configfiles[x]
            self.image_preview = path_name
            str_path_name = str(path_name)
            print(configfiles[x])
            print(path_name)
            @mainthread
            def refresh_path(self):
                self.default_image = path_name

            self.counter_files_indir = str((self.var_len) - (self.counter_ticks)) + ' file(s) remaining...'
            image = cv2.imread(path_name)
            # Resize image and make a copy of the original (resized) image.
            orig = image.copy()

            # Change image color space, if necessary.
            colorSpace = color_space.lower()
            if colorSpace == 'hsv':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif colorSpace == 'ycrcb' or colorSpace == 'ycc':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            elif colorSpace == 'lab':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            else:
                colorSpace = 'bgr'  # set for file naming purposes

            # Keep only the selected channels for K-means clustering.
            if channels_arg != 'all':
                channels = cv2.split(image)
                channelIndices = []
                for char in channels_arg:
                    channelIndices.append(int(char))
                image = image[:, :, channelIndices]
                if len(image.shape) == 2:
                    image.reshape(image.shape[0], image.shape[1], 1)

            # Flatten the 2D image array into an MxN feature vector, where M is
            # the number of pixels and N is the dimension (number of channels).
            reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

            # Perform K-means clustering.
            if clusters < 2:
                print('Warning: num-clusters < 2 invalid. Using num-clusters = 2')
            numClusters = max(2, clusters)
            kmeans = KMeans(n_clusters=numClusters, n_init=2, max_iter=3).fit(reshaped)

            # Reshape result back into a 2D array, where each element represents the
            # corresponding pixel's cluster index (0 to K - 1).
            clustering = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),
                                    (image.shape[0], image.shape[1]))

            # Sort the cluster labels in order of the frequency with which they occur.
            sortedLabels = sorted([n for n in range(numClusters)],
                                  key=lambda x: -np.sum(clustering == x))

            # Initialize K-means grayscale image; set pixel colors based on clustering.
            kmeansImage = np.zeros(image.shape[:2], dtype=np.uint8)
            for i, label in enumerate(sortedLabels):
                kmeansImage[clustering == label] = int(255 / (numClusters - 1)) * i

            # Concatenate original image and K-means image, separated by a gray strip.
            concatImage = np.concatenate((orig,
                                          193 * np.ones((orig.shape[0], int(0.0625 * orig.shape[1]), 3),
                                                        dtype=np.uint8),
                                          cv2.cvtColor(kmeansImage, cv2.COLOR_GRAY2BGR)), axis=1)
            # cv2.imshow('Original vs clustered', concatImage)

            # Construct timestamped output filename and write image to disk.
            fileExtension = '.jpg'
            filename = (datetime.datetime.now().strftime("%Y%m%d%H%M%S") +
                        colorSpace + '_c' + channels_arg + 'n' + str(numClusters) + '.'
                        + fileExtension)
            refresh_path(self)

            # cv2.imwrite(filename, concatImage)
            out_path = (last_path + filename + '.' + fileExtension)
            self.image_preview_result = str(out_path)
            cv2.imwrite(out_path, kmeansImage)

        @mainthread
        def refresh_result(self):
            self.default_image_result = out_path
        refresh_result(self)

        self.counter_ticks = 0

        @mainthread
        def red_mark(self):
            self.IMG_GLOB = "dir"
        red_mark(self)
        self.counter_files_indir = 'Ready Batch of ' + str(self.var_len) + ' files in selected folder'
        print(self.colorspace_arg)

        @mainthread
        def refresh_preview(self):
            self.default_image = AsyncImage(source = 'dir')
            self.default_image_result = AsyncImage(source = 'dir')
        refresh_preview(self)

    def load(self, filename):
        self.dismiss_popup()
        self.path_text_load = filename + '\\'
        return self.path_text_load
        # try:
        # self.ids.image.source = filename[0]
        # except Exception as e:
        # Logger.exception("failure")

    def show_line(self):
        self.path_onscreen_line = str(self.path_text)

    def save(self, path, filename):
        with open(os.path.join(path, filename[0])) as stream:
            self.text_input.text = stream.read()

        self.dismiss_popup()

    def add_widget(self, *args):
        if 'content' in self.ids:
            return self.ids.content.add_widget(*args)
        return super(ShowcaseScreen, self).add_widget(*args)


class LineEllipse1(Widget):
    pass


class ShowcaseApp(App):
    index = NumericProperty(-1)
    current_title = StringProperty()
    time = NumericProperty(0)
    show_sourcecode = BooleanProperty(False)
    screen_names = ListProperty([])
    hierarchy = ListProperty([])

    def build(self):
        self.title = 'Kivy_UI'
        A = 'Image Processing'
        B = 'Results'
        C = 'Quality Check'

        self.icon = 'data/kiwi.ico'

        self.screens = {}
        self.available_screens = sorted({
            A, B, C
        })
        self.screen_names = self.available_screens
        curdir = dirname(__file__)
        self.available_screens = [join(curdir, 'data', 'screens',
                                       '{}.kv'.format(fn).lower()) for fn in self.available_screens]
        self.go_next_screen()



    def on_pause(self):
        return True

    def on_resume(self):
        pass

    def on_current_title(self, instance, value):
        self.root.ids.spnr.text = value

    def go_previous_screen(self):
        self.index = (self.index - 1) % len(self.available_screens)
        screen = self.load_screen(self.index)
        sm = self.root.ids.sm
        sm.switch_to(screen, direction='right')
        self.current_title = screen.name

    def go_next_screen(self):
        self.index = (self.index + 1) % len(self.available_screens)
        screen = self.load_screen(self.index)
        sm = self.root.ids.sm
        sm.switch_to(screen, direction='left')
        self.current_title = screen.name

    def go_screen(self, idx):
        self.index = idx
        self.root.ids.sm.switch_to(self.load_screen(idx), direction='left')

    def load_screen(self, index):
        if index in self.screens:
            return self.screens[index]
        screen = Builder.load_file(self.available_screens[index])
        self.screens[index] = screen
        return screen


Factory.register('Root', cls=ShowcaseScreen)
Factory.register('LoadDialog', cls=LoadDialog)
Factory.register('SaveDialog', cls=SaveDialog)

if __name__ == "__main__":
    ShowcaseApp().run()

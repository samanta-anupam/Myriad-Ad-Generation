from matplotlib.widgets  import RectangleSelector
from helper.config import *

import tensorflow as tf

from ipywidgets import *
from IPython.display import display
from IPython.html import widgets

import os

import cv2
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt
import sys

from tqdm import tnrange, tqdm_notebook
from helper.image_manager import ImageManager
from pprint import pprint

import helper.vae as vae

rect_colors = {'2':'#00FF00', '3':'#0000ff', '4':'#ff0000', '1': '#000000'}
color_tuples = {'2':[0,255,0], '3':[0,0,255], '4':[255,0,0], '1': [0,0,0]}
layers = {'2':2, '3':1, '4':3, '1':0}
layer_names = {'2':'Human', '3':'Body part', '4':'Vehicle', '1':'Text'}


def toggle_selector(event):
    global ax, key_list
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['R', 'r']:
        print(' RectangleSelector reset.')
        ax.cla()
        ax.set_ylim(0,100)
        ax.set_xlim(0,100)
    if event.key in rect_colors.keys():
        ax.set_title('Class:'+layer_names[event.key])
        print('Class', event.key)
        key_list.append(event.key)



CLUSTER_CLASSES_FILE_PATH = '/gpfs/home/asamanta/Myriad-Ad-Generation/models/research/object_detection/class_clusters.txt'

IMAGES_PATH = '/gpfs/scratch/asamanta/dataset/train_images/'
TEXT_DET_OUTPUT = '/gpfs/home/asamanta/Myriad-Ad-Generation/td-output/'
OBJECT_DET_OUTPUT = '/gpfs/home/asamanta/Myriad-Ad-Generation/od-output/'
NUMPY_DATA = '/gpfs/scratch/asamanta/dataset/numpy-data-with-mask-per-layers/'
OUT_TF_DATA = '/gpfs/scratch/asamanta/'

rect_colors = {'2':'#00FF00', '3':'#0000ff', '4':'#ff0000', '1': '#000000'}
color_tuples = {'2':[0,255,0], '3':[0,0,255], '4':[255,0,0], '1': [0,0,0]}
layers = {'2':2, '3':1, '4':3, '1':0}
layer_names = {'2':'Human', '3':'Body part', '4':'Vehicle', '1':'Text'}

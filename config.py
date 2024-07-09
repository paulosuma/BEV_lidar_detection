class Trainingdatapaths:
    def __init__(self):
        self.IMAGE_DATA = "./Datasets/data_object_image_2/training/image_2"
        self.LIDAR_DATA = "./Datasets/data_object_velodyne/training/velodyne"
        self.CALIB_DATA = "./Datasets/data_object_calib/training/calib"
        self.LABEL_DATA = "./Datasets/data_object_label_2/training/label_2"

class Testingdatapaths:
    def __init__(self):
        self.IMAGE_DATA = "./Datasets/data_object_image_2/testing/image_2"
        self.LIDAR_DATA = "./Datasets/data_object_velodyne/testing/velodyne"
        self.CALIB_DATA = "./Datasets/data_object_calib/testing/calib"

# Bounds for the x-axis
X_MIN = 0
X_MAX = 70

# Bounds for the y-axis
Y_MIN = -40
Y_MAX = 40

# Bounds for the z-axis
Z_MIN = -2.5
Z_MAX = 1

# Resolutions
DL = 0.1  # Resolution in length
DW = 0.1  # Resolution in width
DH = 0.1  # Resolution in height

# Image dimensions
IMAGE_HEIGHT = 375
IMAGE_WIDTH = 1240

# set styles for occlusion and truncation
OCC_COL = ['g', 'y', 'r', 'w']
TRUN_STYLE = ['-', '--']

# Define pairs of vertices that need to be connected to form the edges of the box
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # top face edges
    (4, 5), (5, 6), (6, 7), (7, 4),  # bottom face edges
    (0, 4), (1, 5), (2, 6), (3, 7)   # connecting vertical edges
]

OUTPUT_SCALE = 4

REG_MEAN = [-9.16502130e-05,-3.79233648e-04,-1.17230404e-03,3.52957711e-03,2.88674804e-03,8.01448219e-03]
REG_STD = [0.03546629,0.06795792,0.08888607,0.07591444,0.03785192,0.10457838]

DETECTION_THRESH = 0.8
NMS_THRESH = 0.3
N_TOP_CHOSEN_BOXES = 150

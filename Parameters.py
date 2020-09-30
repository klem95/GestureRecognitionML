import numpy as np
import glob2
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
class RecordSettings:
    frameRate = 30
    secondsOfData = 60
    t = frameRate * secondsOfData


class SkeletonData:
    joints = 32
    features = 8
    frame_size = joints * features


class Labeler:
    danceLabels = np.array(['agressive', 'nice', 'big boyy '])




import numpy as np

labels = np.array(['agressive', 'nice', 'big boyy '])

frameRate = 30
secondsOfData = 60
t = frameRate * secondsOfData

joints = 32
features = 8
frame_size = joints * features
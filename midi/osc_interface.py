import live
import random
import time
import numpy as np
from Synth import SynthSetting
from set import set

label1 = SynthSetting('label1', 0, load=True)
label2 = SynthSetting('label2', 1, load=True)
label3 = SynthSetting('label3', 2, load=True)

for play in range(0, 3):

    label1.play()
    for i in range(0, 100):
        label1.lerpParameters(label2.values, i / 100.0)
        time.sleep(0.05)
    time.sleep(1)
    label1.stop()
    label2.play()
    for i in range(0, 100):
        label2.lerpParameters(label1.values, i / 100.0)
        time.sleep(0.05)
    time.sleep(1)
    label2.stop()
    label3.play()
    for i in range(0, 100):
        label1.lerpParameters(label3.values, i / 100.0)
        time.sleep(0.05)
    label3.stop()

track = set.tracks[0]


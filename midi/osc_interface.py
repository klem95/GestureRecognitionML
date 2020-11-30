import live
import random
import time
import numpy as np
from .Synth import SynthSetting
from .set import set
from . import tools



def savePreset():
    SynthSetting('label3', 0, save=True)




class Player:
    def __init__(self, _labelNames):
        self.labelPredictions = None
        self.isPlaying = False
        self.labelNames = _labelNames
        self.baseSynth = SynthSetting('label3', 0, load=True)
        self.currentSynth = 0
        self.synths = [
            SynthSetting('label1', 0, load=True),
            SynthSetting('label2', 1, load=True),
            SynthSetting('label3', 2, load=True),
        ]



    def play(self):
        self.isPlaying = True

    def updatePredictions(self, labelPredictions):
        self.labelPredictions = labelPredictions
        self.currentSynth = self.labelNames.index(labelPredictions[0][0])
        print(labelPredictions[0])
        # self.currentSynth = labelPredictions

    def clock(self):
        # self.synths[]
        self.baseSynth.lerpParameters()
        print(self.labelPredictions[0])
        pass




def loadAndPlay():
    label1 = SynthSetting('label1', 0, load=True)
    label2 = SynthSetting('label3', 1, load=True)
    label3 = SynthSetting('label2', 2, load=True)

    for play in range(0, 3):
        label1.play()
        for i in range(0, 16):
            label1.lerpParameters(label2.values, i / 16.0)
            tools.waitForBeats(1)
        label2.play()
        tools.waitForBeats(8)
        label1.stop()
        tools.waitForBeats(8)
        for i in range(0, 16):
            label2.lerpParameters(label3.values, i / 16.0)
            tools.waitForBeats(1)
        label3.play()
        tools.waitForBeats(8)
        label2.stop()
        tools.waitForBeats(4)
        for i in range(0, 100):
            label3.lerpParameters(label1.values, i / 100.0)
            time.sleep(0.05)
        label3.stop()




# savePreset()



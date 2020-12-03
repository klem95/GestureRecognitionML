import time
import numpy as np
from .Synth import SynthSetting
from . import tools
import os
import sys


startIndex = 29
synthAmount = 28


def globalNameSynth(i):
    return 'SYNTH_' + str(i)

def globalNameRack(i):
    return 'RACK_' + str(i)

def savePreset():
    print('saving presets...')
    print(os.path.dirname(os.path.realpath(sys.argv[0])))

    for i in range(0, synthAmount):
        print('loading + ' + str(i))
        SynthSetting(globalNameSynth(i), i + startIndex, save=True, device=0)
        SynthSetting(globalNameRack(i), i + startIndex, save=True, device=1)





class Player:
    def __init__(self):
        self.labelPredictions = None
        self.isPlaying = False
        self.baseSynth = SynthSetting(globalNameSynth(0), 28, load=True, device=0)
        self.baseRack= SynthSetting(globalNameSynth(0), 28, load=True, device=1)
        self.currentSynth = 0

        self.synths = [SynthSetting(globalNameSynth(i), i + startIndex, load=True, device=0) for i in range(0, synthAmount)]
        self.racks = [SynthSetting(globalNameRack(i), i + startIndex, load=True, device=1) for i in range(0, synthAmount)]

        print(self.synths)


    def weightedAverage(self, x, weights):
        y = np.sum([x[i] * weights[i] for i in range(len(x))]) / np.sum(weights)
        return round(y, 1)



    def lerpAllLabels(self, predictions):
        parameters = self.baseSynth.parameters
        weightedParameters = np.array([])

        for i in range(len(parameters)): # 200~ params
            paramForSynths = np.array([])
            weights = np.array([])
            for synth in self.synths:
                np.append(paramForSynths, float(synth.values[i][3]))
                print(paramForSynths)

            for weight in predictions:
                np.append(weights, weight[1])
            print('weights')
            print(weights)
            np.append(weightedParameters, self.weightedAverage(paramForSynths, weights))
        print(weightedParameters)
        return weightedParameters



    def play(self):
        self.isPlaying = True

    def updatePredictions(self, labelPredictions):
        self.lerpAllLabels(labelPredictions)
        #print(labelPredictions[0])
        self.labelPredictions = labelPredictions
        # self.currentSynth = self.labelNames.index(labelPredictions[0][0])
        # self.currentSynth = labelPredictions

    def clock(self):
        # self.synths[]
        # print(self.labelPredictions[0])
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







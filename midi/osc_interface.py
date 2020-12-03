import time
import numpy as np
from .Synth import SynthSetting
from . import tools
import os
import sys


startIndex = 29
synthAmount = 28
bpms = [
    20, 48, 20, 100, 67, 60, 72, 37, 94, 147, 82, 63, 90, 90, 105, 73, 86, 100, 94, 36, 80, 40, 20, 41, 37, 65, 22, 48
]

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
        self.baseRack= SynthSetting(globalNameRack(0), 28, load=True, device=1)
        self.currentSynth = 0

        self.synths = [SynthSetting(globalNameSynth(i), i + startIndex, load=True, device=0) for i in range(0, synthAmount)]
        self.racks = [SynthSetting(globalNameRack(i), i + startIndex, load=True, device=1) for i in range(0, synthAmount)]

        self.clips = [SynthSetting('CLIP_' + str(i), i) for i in range(0, synthAmount)]

        print(self.synths)


    def weightedAverage(self, x, weights):
        y = np.sum([x[i] * weights[i] for i in range(x.shape[0])]) / np.sum(weights)
        return y



    def lerpAllParameters(self, device, predictions):
        values = device.values
        weightedParameters = np.empty(len(values), dtype=float)

        for i in range(len(values)): # 200~ params
            parameterPerSynth = np.empty(len(self.synths), dtype=float)
            weights = np.empty(len(self.synths), dtype=float)
            for s in range(len(self.synths)):
                parameterPerSynth[s] = float(self.synths[s].values[i][3])
            for w in range(len(predictions)):
                weights[w] = predictions[w][1]
            weightedParameters[i] = self.weightedAverage(parameterPerSynth, weights)
        return list(weightedParameters)



    def play(self):
        self.isPlaying = True

    def updatePredictions(self, labelPredictions):
        newParamsSynth = self.lerpAllParameters(self.baseSynth, labelPredictions)
        newParamsRack = self.lerpAllParameters(self.baseRack, labelPredictions)

        for i in range(len(newParamsSynth)):
            self.baseSynth.setParameter(i, newParamsSynth[i])
        for i in range(len(newParamsRack)):
            self.baseRack.setParameter(i, newParamsRack[i])
        self.baseRack.setParameter(len(newParamsRack) - 1, 0)

        weights = np.empty(len(self.synths), dtype=float)
        for w in range(len(labelPredictions)):
            weights[w] = labelPredictions[w][1]
        tools.set.tempo = self.weightedAverage(np.asarray(bpms), weights)



        predictions = []
        for i in range(len(labelPredictions)):
            predictions.append(labelPredictions[i][1])
        maxPrediction = max(predictions)
        maxPredictionIndex = predictions.index(maxPrediction)

        #
        # for i in range(len(self.clips)):
        #     if(maxPredictionIndex != i):
        #         self.clips[i].stop()
        if(maxPrediction > 0.95):
            self.clips[maxPredictionIndex].play()
            print(maxPredictionIndex)



        #        if(bestPrediction)

        #self.baseSynth.setParameters(None, newParams)
        #print(labelPredictions[0])
        # self.labelPredictions = labelPredictions
        # self.currentSynth = self.labelNames.index(labelPredictions[0][0])
        # self.currentSynth = labelPredictions

    def updateDynamicParameters(self, dynamicParams):
        print(dynamicParams)
        self.baseRack.setParameter(1, min(127, dynamicParams[2][1]))
        self.baseRack.setParameter(5, min(127, dynamicParams[3][1]))
        self.baseRack.setParameter(6, min(127, dynamicParams[3][1]))
        pass

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

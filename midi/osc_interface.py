import time
import numpy as np
from .Synth import SynthSetting
from . import tools
import os
import sys
from datetime import datetime
import time
sys.path.append('..')
from .Skeleton_structure import Skeleton
from .SYNTH_PARAM import SYNTH
startIndex = 2
synthAmount = 28
bpms = [
    20, 48, 20, 100, 67, 60, 72, 37, 94, 147, 82, 63, 90, 90, 105, 73, 86, 100, 94, 36, 80, 40, 20, 41, 37, 65, 22, 48
]

def globalNameSynth(i):
    return 'SYNTH_' + str(i)

def globalNameRack(i):
    return 'RACK_' + str(i)

def globalNameClip(i):
    return 'LABEL_' + str(i + 1) + '_START'


def savePreset():
    print('saving presets...')
    print(os.path.dirname(os.path.realpath(sys.argv[0])))

    for i in range(0, synthAmount):
        print('loading + ' + str(i))
        SynthSetting(globalNameSynth(i), i + startIndex, save=True, device=0)
        SynthSetting(globalNameRack(i), i + startIndex, save=True, device=1)


synthMaxParameters = None


class Player:
    def __init__(self):

        self.labelPredictions = None
        self.isPlaying = False
        self.baseSynth = SynthSetting(globalNameSynth(0), 1, load=True, device=0)
        self.baseRack= SynthSetting(globalNameRack(0), 1, load=True, device=1)
        self.baseRack= SynthSetting(globalNameRack(0), 1, load=True, device=1)
        self.currentSynth = 0

        global synthMaxParameters
        global rackMaxParameters
        synthMaxParameters = self.baseSynth.getMaximumValues()
        rackMaxParameters = self.baseRack.getMaximumValues()
        self.parameterModifiers = [
            parameterModifier(170, Skeleton.WRIST_R_VEL, 0.5, 0),      # 1
            # parameterModifier(, [Seleton.WRIST_R_VEL]7, 1),      # 2
            parameterModifier(34, Skeleton.WRIST_R_VEL, 0.7, 2),      # 3
            parameterModifier(170, Skeleton.WRIST_R_VEL, 0.3, 3),      # 4
            parameterModifier(148, Skeleton.WRIST_R_VEL, 1.0, 4),      # 5
            parameterModifier(170, Skeleton.WRIST_R_VEL, -0.5, 5),      # 6
            parameterModifier(170, Skeleton.WRIST_R_VEL, 0.5, 6),      # 7
            parameterModifier(20, Skeleton.WRIST_R_VEL, -1.0, 7),      # 8
            parameterModifier(88, Skeleton.WRIST_R_VEL, 0.1, 8),      # 9
            parameterModifier(61, Skeleton.WRIST_R_VEL, 1.0, 9),      # 10
            parameterModifier(120, Skeleton.WRIST_R_VEL, 1.0, 10),      # 11
            parameterModifier(170, Skeleton.WRIST_R_VEL, 0.7, 11),      # 12
            parameterModifier(20, Skeleton.WRIST_R_VEL, -0.5, 12),      # 13
            parameterModifier(170, Skeleton.WRIST_R_VEL, 0.3, 13),      # 14
            parameterModifier(137, Skeleton.WRIST_R_VEL, 1.0, 14),      # 15
            parameterModifier(58, Skeleton.WRIST_R_VEL, 0.2, 15),      # 16
            parameterModifier(20, Skeleton.WRIST_R_VEL, -1.0, 16),      # 17
            parameterModifier(170, Skeleton.WRIST_R_VEL, -0.6, 17),      # 18
            parameterModifier(120, Skeleton.WRIST_R_VEL, -1.0, 18),      # 19
            parameterModifier(120, Skeleton.WRIST_R_VEL, 1.0, 19),      # 20
            parameterModifier(120, Skeleton.WRIST_R_VEL, 1.0, 20),      # 21
            parameterModifier(120, Skeleton.WRIST_R_VEL, 1.0, 21),      # 22
            parameterModifier(2, Skeleton.WRIST_R_VEL, 0.8, 22),      # 23
            parameterModifier(170, Skeleton.WRIST_R_VEL, 0.3, 23),      # 24
            parameterModifier(171, Skeleton.WRIST_R_VEL, 1.0, 24),      # 25
            #parameterModifier(, [Seleton.WRIST_R_VEL]7, 25),      # 26
            parameterModifier(171, Skeleton.WRIST_R_VEL, 1.0, 26),      # 27
            parameterModifier(170, Skeleton.WRIST_R_VEL, 0.3, 27),      # 28

        ]

        self.rackModifiers = []
        for i in range(28):
            self.rackModifiers.append(parameterModifier(5, Skeleton.ANGLE_R_WING, 0.8, i, True))
            self.rackModifiers.append(parameterModifier(6, Skeleton.ANGLE_L_WING, 0.8, i, True))
            # self.parameterModifiers.append(parameterModifier(170, Skeleton.PELVIS_POS_Y, -2, i))
            self.parameterModifiers.append(parameterModifier(5, Skeleton.PELVIS_POS_X, -1, i))
            self.parameterModifiers.append(parameterModifier(6, Skeleton.PELVIS_POS_X, 21, i))
            pass



        self.synths = [SynthSetting(globalNameSynth(i), 1, load=True, device=0) for i in range(0, synthAmount)]
        self.racks = [SynthSetting(globalNameRack(i), 1, load=True, device=1) for i in range(0, synthAmount)]
        self.clipPlaying = None
        self.ClipPlayer = SynthSetting('CLIP_TRACK', 0, scanClipNames=True)
        self.lastInputTimeAgo = time.time()
        self.newParamsSynth = None
        self.newParamsRack = None
        self.labelPredictions = None

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


    def setParameters(self):
        for i in range(len(self.newParamsSynth)):
            self.baseSynth.setParameter(i, self.newParamsSynth[i])
        for i in range(len(self.newParamsRack)):
            self.baseRack.setParameter(i, self.newParamsRack[i])
        self.baseRack.setParameter(len(self.newParamsRack) - 1, 0)

        pass

    def updatePredictions(self, labelPredictions):
        self.labelPredictions = labelPredictions
        deltaTime = time.time() - self.lastInputTimeAgo
        self.lastInputTimeAgo = time.time()
        if deltaTime > 0.5:
            tools.set.volume = 1 / (deltaTime / 0.5)
        else:
            tools.set.volume = 1

        self.newParamsSynth = self.lerpAllParameters(self.baseSynth, labelPredictions)
        self.newParamsRack = self.lerpAllParameters(self.baseRack, labelPredictions)

        predictions = []
        for i in range(len(labelPredictions)):
            predictions.append(labelPredictions[i][1])
        maxPrediction = max(predictions)
        maxPredictionIndex = predictions.index(maxPrediction)

        print(labelPredictions[maxPredictionIndex])
        weights = np.empty(len(self.synths), dtype=float)
        for w in range(len(self.labelPredictions)):
            weights[w] = self.labelPredictions[w][1]

        tempo = self.weightedAverage(np.asarray(bpms), np.exp(np.exp(weights) - 1) - 1)
        tools.set.tempo = tempo

        if(maxPrediction > 0.85):
            if self.clipPlaying != maxPredictionIndex:
                print('Hit threshold: ')
                print(globalNameClip(maxPredictionIndex))
                print(labelPredictions[maxPredictionIndex])
                self.clipPlaying = maxPredictionIndex
                self.ClipPlayer.play(globalNameClip(maxPredictionIndex))



    def updateDynamicParameters(self, dynamicParams):
        # print(dynamicParams)
        for i in range(len(self.newParamsRack)):
            rackParam = self.newParamsRack[i]
            # print(rackParam)

        for parameterModifier in self.parameterModifiers:
            self.newParamsSynth = parameterModifier.modifyParameters(self.newParamsSynth, dynamicParams, self.labelPredictions)

        for rackModifier in self.rackModifiers:
            self.newParamsRack = rackModifier.modifyParameters(self.newParamsRack, dynamicParams, self.labelPredictions)

        # tools.set.set_master_volume((dynamicParams[Skeleton.PELVIS_POS_Z][1] * 1.3))
        tools.set.set_master_pan(-dynamicParams[Skeleton.PELVIS_POS_X][1] * 3)
        self.setParameters()
        pass

    def clock(self):
        # self.synths[]
        # print(self.labelPredictions[0])
        pass




class parameterModifier():
    def __init__(self, parameterToModify, parameterModifiedBy, modifyAmount, label, isRack=False):
        self.modifyAmount = modifyAmount
        self.parameterToModify = parameterToModify
        self.parameterModifiedBy = parameterModifiedBy
        self.label = label
        global synthMaxParameters
        global rackMaxParameters
        if isRack:
            self.maxValue = rackMaxParameters[parameterToModify]
        else:
            self.maxValue = synthMaxParameters[parameterToModify]

    def modifyParameters(self, parameters, dynamicParameters, predictions):
        labelModifier = predictions[self.label]

        mod = 0


        if(isinstance(self.parameterModifiedBy, list)):
            mods = []
            for joint in self.parameterModifiedBy:
                mods.append(dynamicParameters[joint][1] * labelModifier[1])
            mod = sum(mods) / len(mods)
        else:
            mod = dynamicParameters[self.parameterModifiedBy][1] * labelModifier[1]



        modification = self.modifyAmount * mod * self.maxValue
        newParam = parameters[self.parameterToModify] + modification
        parameters[self.parameterToModify] = newParam
        # print(dynamicParameters[3])
        if modification > 0.5 * mod * self.maxValue:
            pass
            # print()
            # print('for label %s' %predictions[self.label][0], labelModifier[1])
            # print('modifying param ' + SYNTH[self.parameterModifiedBy] +' (' +str(self.parameterToModify) + '), using input ' + str(self.parameterModifiedBy) + ', value ' + str(modification))
            # print('Max value ' + str(self.maxValue))
            # print('    value ' + str(modification))
        return parameters




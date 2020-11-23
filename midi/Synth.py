import live
import tools
from set import set
track = 4
device = 0
Synth = set.tracks[track]
Device = Synth.devices[device]


class SynthSetting():
    def __init__(self, _name, _track, save=False, load=False):
        self.parameters = Synth.devices[0].parameters
        self.name = _name
        self.track = _track
        self.values = []
        if(save):
            self.getParameters()
            self.saveParameters()
        if(load):
            self.loadParameters()
            self.setParameters()


    def setParameter(self, id, value):
        Device.parameters[id].value = value

    def getParameter(self, id):
        return Device.parameters[id].value

    def getParameters(self):
        for i in range(0, len(self.parameters)):
            self.values.append(self.getParameter(i))

    def setParameters(self):
        for i in range(0, len(self.parameters)):
            self.setParameter(i, float(self.values[i][3]))

    def saveParameters(self):
        tools.saveModel(self.values, self.name)

    def loadParameters(self):
        self.values = tools.loadModel(self.name)

    def lerpParameters(self, otherParameters, t):
        for i in range(0, len(self.parameters)):
            param = float(self.values[i][3])
            otherParam = float(otherParameters[i][3])
            lerpedParam = tools.lerp(param, otherParam, t)
            self.setParameter(i, lerpedParam)
    def play(self):
        print('playing ' + self.name)
        set.tracks[self.track].clips[1].play()
    def stop(self):
        print('stopping ' + self.name)
        set.tracks[self.track].stop()

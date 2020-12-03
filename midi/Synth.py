print('Synth.py: importing tools...')
from . import tools
print('Synth.py: importing set...')
from .set import set


class SynthSetting():
    def __init__(self, _name, track = 4, save=False, load=False, device=0):
        print('init synth ' + _name)
        self.track = track
        self.Track = set.tracks[self.track]

        self.Device = self.Track.devices[device]
        self.parameters = self.Track.devices[device].parameters

        self.name = _name
        self.values = []
        if(save):
            self.getParameters()
            self.saveParameters()
        if(load):
            self.loadParameters()
            self.setParameters()


    def setParameter(self, id, value):
        self.Device.parameters[id].value = value

    def getParameter(self, id):
        return self.Device.parameters[id].value

    def getParameters(self):
        for i in range(0, len(self.parameters)):
            self.values.append(self.getParameter(i))

    def setParameters(self, parameters = None, values = None):
        if parameters == None:
            parameters = self.parameters
        if values == None:
            values = self.values


        for i in range(0, len(parameters)):
            self.setParameter(i, float(values[i][3]))

    def saveParameters(self):
        tools.saveModel(self.values, self.name)

    def loadParameters(self):
        print('loading: ' + self.name )
        self.values = tools.loadModel(self.name)
        if self.values == False:
            raise Exception("File not found: " + self.name)

    def lerpParameters(self, otherParameters, t):
        for i in range(0, len(self.parameters)):
            param = float(self.values[i][3])
            otherParam = float(otherParameters[i][3])
            lerpedParam = tools.lerp(param, otherParam, t)
            self.setParameter(i, lerpedParam)

    def play(self):
        print('playing ' + self.name)
        self.Track.clips[1].play() # self.Track[self.track].clips[1].play()

    def stop(self):
        print('stopping ' + self.name)
        self.Track.stop()

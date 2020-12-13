print('Synth.py: importing tools...')
from . import tools
print('Synth.py: importing set...')
from .set import set


class SynthSetting():
    def __init__(self, name, track = 4, save=False, load=False, device=0, scanClipNames = False,):
        print('INIT SETTING ' + name)
        self.track = track
        self.Track = set.tracks[self.track]

        self.isPlaying = False,

        self.Device = None
        if (save or load):
            self.Device = self.Track.devices[device]
            self.parameters = self.Track.devices[device].parameters

        self.name = name
        self.values = []
        self.maxValues = []
        if save:
            self.getParameters()
            self.saveParameters()
        if load:
            self.loadParameters()
            self.setParameters()


        self.clipList = []
        if scanClipNames:
            # self.Track.scan_clip_names() # self.Track[self.track].clips[1].play()
            for i in range(len(self.Track.clips)):
                if self.Track.clips[i] != None and self.Track.clips[i].name != '':
                    self.clipList.append((self.Track.clips[i].name, i))
            print(self.clipList)

    def getMaximumValues(self):
        maxParams = []
        for param in self.parameters:
            maxParams.append(param.maximum)
        return maxParams

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
        print('loading: ' + self.name)
        self.values = tools.loadModel(self.name)
        if self.values == False:
            raise Exception("File not found: " + self.name)

    def lerpParameters(self, otherParameters, t):
        for i in range(0, len(self.parameters)):
            param = float(self.values[i][3])
            otherParam = float(otherParameters[i][3])
            lerpedParam = tools.lerp(param, otherParam, t)
            self.setParameter(i, lerpedParam)

    def play(self, labelName):
        self.isPlaying = True
        for clip in self.clipList:
            if clip[0] == labelName:
                print('playing ' + str())
                self.Track.clips[clip[1]].play() # self.Track[self.track].clips[1].play()
                break



    def stop(self):
        if self.isPlaying == True:
            print(self.isPlaying)
            self.isPlaying = False
            print('stopping ' + self.name)
            set.stop_track(track_index=self.track)

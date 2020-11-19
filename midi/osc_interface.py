import live
import random
import time
import numpy as np
from numpy import load, save, genfromtxt

#------------------------------------------------------------------------
# Scan the set's contents and set its tempo to 110bpm.
#------------------------------------------------------------------------
set = live.Set()
set.scan(scan_clip_names = True, scan_devices = True)
set.tempo = 110.0
print(set.tempo)

#
# for t in range (100, 120):
#     set.tempo = t
#     time.sleep(0.05)


def saveModel(object, name):
    model_json = object.to_json()
    with open('presets/' + name + '.json', "w") as json_file:
        json_file.write(model_json)
    print("Saved preset to disk")

def loadModel(name):
    json_file = open('presets/' + name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    return loaded_model_json



standardParameters = [

]


Synth = set.tracks[4]

class SynthSetting():
    def __init__(self, _name, save=False, load=False):
        self.parameters = Synth.devices[0].parameters
        self.name = _name
        self.values = []
        if(save):
            self.getParameters()
            self.saveParameters()
        if(load):
            self.loadParameters()


    def getParameters(self):
        for p in self.parameters:
            print('Appending: ' + p.value)
            self.values.append(p.value[3])
            print(p.value[3])

    def setParameters(self):
        for i in range(0, len(self.parameters)):
            p = self.parameters[i]
            print('setting: ' + p.value)
            p.value = self.values[i]
            print(p.value)

    def saveParameters(self):
        saveModel(self.values, self.name)

    def loadParameters(self):
        self.values = loadModel(self.name)





synth1 = SynthSetting(save=True)

print(dir(Synth))










#------------------------------------------------------------------------
# Each Set contains a list of Track objects.
#------------------------------------------------------------------------
track = set.tracks[0]

for label in set.tracks:
    print("Track name %s" % label.name)


#------------------------------------------------------------------------
# Each Track contains a list of Clip objects.
#------------------------------------------------------------------------
clip = track.clips[1]
print("Clip name %s, length %d beats" % (clip.name, clip.length))
clip.play()

#------------------------------------------------------------------------
# We can determine our internal timing based on Live's timeline using
# Set.wait_for_next_beat(), and trigger clips accordingly.
#------------------------------------------------------------------------
set.wait_for_next_beat()
clip.get_next_clip().play()

#------------------------------------------------------------------------
# Now let's modulate the parameters of a Device object.
#------------------------------------------------------------------------
device = track.devices[0]
parameter = random.choice(device.parameters)
parameter.value = random.uniform(parameter.minimum, parameter.maximum)

from numpy import load, save, genfromtxt
import os
print('tools.py: importing set')
from .set import set


def lerp(a, b, f):
    return a + f * (b - a)



def saveModel(object, name):
    print("Saving...")
    save(os.path.dirname(os.path.realpath(__file__)) + '/presets/' + name +  '.npy', object)
    print("Saved %s preset to disk", name)

def loadModel(name):
    try:
        print('Loading preset')
        npObject = load('GestureRecognitionML/midi/presets/' + name + '.npy', allow_pickle=True)
        return npObject
    except:
        print('No preset named: ' + name)
        return False


def waitForBeats(beats):
    for i in range(0, beats):
        print('waiting for beat ' + str(i + 1) + ' of ' + str(beats))
        set.wait_for_next_beat()

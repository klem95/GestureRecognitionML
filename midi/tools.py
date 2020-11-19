from numpy import load, save, genfromtxt
import live

from set import set


def lerp(a, b, f):
    return a + f * (b - a)



def saveModel(object, name):
    print("Saving...")
    save('midi/presets/' + name +  '.npy', object)
    print("Saved %s preset to disk", name)

def loadModel(name):
    try:
        print('Loading preset')
        npObject = load('midi/presets/' + name + '.npy', allow_pickle=True)
        return npObject
    except:
        print('No preset named: ' + name)
        return False


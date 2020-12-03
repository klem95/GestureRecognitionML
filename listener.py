
# from midi.osc_interface import Player

from midi.osc_interface import Player
import sys
sys.path.append('..')
from Networking import Receiver

class Listener():
    def __init__(self):
        self.liveInterface = Player()
        self.Receiver = Receiver()
        while True:
            result = self.Receiver.listen()
            prediction = result[0]
            dynamicParams = result[1]
            self.liveInterface.updatePredictions(prediction)
            self.liveInterface.updateDynamicParameters(dynamicParams)


listener = Listener()


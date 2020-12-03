
# from midi.osc_interface import Player

import sys
sys.path.append('..')
from Networking import UDP

class Listener():
    def __init__(self):
        self.UDP = UDP()
        while True:
            print(self.UDP.receiver())
        self.liveInterface = Player()
        self.testManager = None
        self.liveInterface.play()


listener = Listener()
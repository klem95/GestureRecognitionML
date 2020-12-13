
# from midi.osc_interface import Player

from midi.osc_interface import Player
import sys
import time
sys.path.append('..')
from Networking import Receiver


start_time = time.time()
x = 1 # displays the frame rate every 1 second
counter = 0


class Listener():
    def __init__(self):
        self.liveInterface = Player()
        self.Receiver = Receiver()
        ########################


        while True:
            result = self.Receiver.listen()
            prediction = result[0]
            dynamicParams = result[1]
            global counter
            global start_time
            counter += 1
            if (time.time() - start_time) > x:
                print("FPS: ", counter / (time.time() - start_time))
                counter = 0
                start_time = time.time()
            self.liveInterface.updatePredictions(prediction)
            self.liveInterface.updateDynamicParameters(dynamicParams)


listener = Listener()


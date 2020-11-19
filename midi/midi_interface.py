# /Users/macbook/miniconda3/envs/AIP/bin/pip install rtmidi
from rtmidi import MidiMessage, RtMidiIn, RtMidiOut
import time
import midi_tools

midiout = RtMidiOut()
midiout.openPort(2)

midiin = RtMidiIn()
midiin.openPort(2)

midi_tools.print_ports(midiin)
midi_tools.print_ports(midiout)

class MidiConnection:
    def __init__(self, _channel, _DEBUG=False, _port = 0):
        self.port = _port
        self.channel = _channel
        self.DEBUG = _DEBUG

    def output(self, m):
        midiout.sendMessage(m)
        if(self.DEBUG):
            midi_tools.print_message(m)

    def sendNoteSignal(self, note, vel, wait=.1):
        m = MidiMessage.noteOn(self.channel, note, vel)
        self.output(m)
        time.sleep(wait)
        m = MidiMessage.noteOff(self.channel, note)
        self.output(m)

    def listenForInput(self):
        while True:
            m = midiin.getMessage(200)  # some timeout in ms
            if m:
                midi_tools.print_message(m)



keyboardTrigger = MidiConnection(0, False, _port=2)
for i in range(12, 12 * 8):
    keyboardTrigger.sendNoteSignal(i, 100, .02)


keyboardTrigger.listenForInput()


midiin.closePort()
print('quit')








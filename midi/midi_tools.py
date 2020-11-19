
types = [
    'isActiveSense', 'isAftertouch', 'isAllNotesOff', 'isAllSoundOff', 'isChannelPressure', 'isController',
    'isNoteOff', 'isNoteOn', 'isNoteOnOrOff', 'isPitchWheel', 'isProgramChange', 'isSysEx'
]

def checkType (midi):
    for func in types:
        method_to_call = getattr(midi, func)
        result = method_to_call()
        if(result):
            print(' - ', func, result)

def print_message(midi):
    if midi.isNoteOn():
        print('ON: ', midi.getMidiNoteName(midi.getNoteNumber()), midi.getVelocity())
    elif midi.isNoteOff():
        print('OFF:', midi.getMidiNoteName(midi.getNoteNumber()))
    elif midi.isController():
        print('CONTROLLER', midi.getControllerNumber(), midi.getControllerValue())
    elif midi.isSysEx():
        print('SYS EX', midi.getControllerNumber(), midi.getControllerValue())
        print(midi.getSysExData())

    else:
        print('unknown msg')
        checkType(midi)
        print(midi.getChannel())
        print(midi.getControllerName(midi.getControllerNumber()))
        print(midi.getControllerValue())
        # print(dir(midi))

def print_ports(device):
    ports = range(device.getPortCount())
    if ports:
        for i in ports:
            print('port: ' + str(i) + ': MIDI:', device.getPortName(i))
    else:
        print('NO MIDI PORTS!')



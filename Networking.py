import socket
import ast
import numpy as np


class UDP:
    def __init__(self):
        self.UDP_IP = "127.0.0.1"  # Should change to something like 192.168.x.x
        self.UDP_port = 5005

    def sender(self, predictions=[], skeleton_data=[]):
        print("Sending to UDP target IP: %s" % self.UDP_IP)
        print("Sending to UDP target port: %s" % self.UDP_port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        composed_list = [predictions, skeleton_data]
        sock.sendto(bytes(str(composed_list).encode('utf-8')), (self.UDP_IP, self.UDP_port))

    def receiver(self):
        print("Listening to UDP target IP: %s" % self.UDP_IP)
        print("Listening to UDP target port: %s" % self.UDP_port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.UDP_IP, self.UDP_port))

        while True:
            data, addr = sock.recvfrom(1024)
            print("received message: %s" % data)
            res = ast.literal_eval(data.decode("utf-8"))
            return res

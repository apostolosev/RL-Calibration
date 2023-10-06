import time
import serial


class Nema17:
    def __init__(self, port="/dev/ttyACM0", baudrate=155200):
        self.port = port
        self.baudrate = baudrate
        self.arduino = serial.Serial(port=port, baudrate=baudrate)
        msg = self.arduino.readline()
        print(msg.decode())

    def set_port(self, port):
        self.port = port
        self.arduino.port = port

    def set_baudrate(self, baudrate):
        self.baudrate = baudrate
        self.arduino.baudrate = baudrate

    def move(self, rev1=0, rev2=0, rev3=0):
        data = "<" + str(rev1) + "," + str(rev2) + "," + str(rev3) + ">"
        self.arduino.write(data.encode())
        msg = self.arduino.readline()
        # print(msg.decode())


def main():
    nema17 = Nema17()
    begin = time.time()
    nema17.move(1000, 1000, 1000)
    end = time.time()
    print("Elapsed time: {}".format(round(1000 * (end - begin))))


if __name__ == "__main__":
    main()

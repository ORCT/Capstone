import serial
from collections import deque

def interact_ser(_str, _ard):
    _ard.write(_str.encode())
    if _str[-1] == '`':
        tmp = ""
        while tmp == "":
            tmp = _ard.readline()
        print(tmp.decode())
        return tmp
    


if __name__ == "__main__":
    port = 'COM12'  # 변동가능
    ard = serial.Serial(port, 9600)

    tmp_list = deque([])

    while True:
        tmp = input()
        tmp_list.append(tmp)
        if tmp == '-':
            break
        else:
            interact_ser(tmp_list.popleft(), ard)

    ard.close()

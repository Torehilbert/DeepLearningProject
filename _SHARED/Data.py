import threading
import queue
import os
import time
import socket
import struct


class Data:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = []

    def add_data(self, data):
        self.lock.acquire()
        self.data.extend(data)
        self.lock.release()

    def extract_data(self):
        self.lock.acquire()
        dataCopy = self.data.copy()
        self.data = []
        self.lock.release()
        return dataCopy

    def terminate(self):
        if(self.file is not None):
            self.file.close()


class DataMaster:
    def __init__(self, n, filepaths=None, update_rate=0.25, udp=True):
        self.n = n
        self.filepaths = filepaths
        self.udp_on = udp

        self.qs = []
        for i in range(n):
            self.qs.append(queue.Queue())

        self.update_rate = update_rate
        self.remover_thread = threading.Thread(target=self._thread_main, args=(filepaths,))
        self.remover_thread.start()
        self.thread_active = True

    def add_data(self, element, id):
        self.qs[id].put(element)

    def add_data_array(self, data, ids=None):
        if(ids is None):
            ids = [i for i in range(len(data))]

        for i in range(len(data)):
            self.qs[ids[i]].put(data[i])

    def close(self):
        self.thread_active = False

    def _thread_main(self, filepaths=[]):
        # open files
        self.files = [None] * self.n
        for i in range(len(filepaths)):
            path = filepaths[i]
            if(path is not None):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self.files[i] = open(path, 'w')

        # open socket
        if(self.udp_on):
            self.IP = "127.0.0.1"
            self.PORT = 12000
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # main thread loop
        while(self.thread_active):
            data = self._thread_extract_data()
            if(self.filepaths is not None):
                self._thread_log_to_files(data)
            if(self.udp_on):
                self._thread_send_via_udp(data)
            time.sleep(self.update_rate)

        for i in range(len(self.files)):
            if(self.files[i] is not None):
                self.files[i].close()

    def _thread_extract_data(self):
        max_count = 1000
        data = []
        for q in self.qs:
            elements = []
            try:
                for i in range(max_count):
                    element = q.get_nowait()
                    elements.append(element)
            except queue.Empty:
                pass
            finally:
                data.append(elements)
        return data

    def _thread_log_to_files(self, all_data):
        for i in range(self.n):
            f = self.files[i]
            if(f is None):
                continue
            data = all_data[i]
            cont_string = ''
            for j in range(len(data)):
                cont_string += str(data[j]) + "\n"
            if(len(cont_string) > 0):
                f.write(cont_string)

    def _thread_send_via_udp(self, all_data):
        for i in range(self.n):
            if(len(all_data[i]) > 0):
                data = [i] + all_data[i]
                bytes_data = struct.pack("%df" % len(data), *data)
                self.socket.sendto(bytes_data, (self.IP, self.PORT))


class DataUDPReceiver:
    def __init__(self, n, IP="127.0.0.1", PORT=12000):
        self.n = n
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ip = IP
        self.port = PORT
        self.socket.bind((IP, PORT))

        self.qs = []
        for i in range(n):
            self.qs.append(queue.Queue())

        self.thread_active = True
        self.thread = threading.Thread(target=self._thread_listener)
        self.thread.start()

    def fetch(self, channel):
        max_count = 10000
        q = self.qs[channel]

        data = []
        try:
            for i in range(max_count):
                element = q.get_nowait()
                data.append(element)
        except queue.Empty:
            pass
        finally:
            return data

    def fetch_all(self):
        all_data = []
        for i in range(self.n):
            all_data.append(self.fetch(channel=i))
        return all_data

    def _thread_listener(self):
        while(self.thread_active):
            data, addr = self.socket.recvfrom(1024)
            pack = struct.unpack('%df' % (len(data) // 4), data)
            channel = int(pack[0])
            for val in pack[1:]:
                self.qs[channel].put(val)


if __name__ == "__main__":
    folder = os.path.dirname(os.path.realpath(__file__))
    fpaths = [os.path.join(folder, 'testset1.csv'), None, os.path.join(folder, 'testset2.csv')]
    update_rate = 0.25

    DM = DataMaster(3, filepaths=fpaths, update_rate=update_rate)

    print("Sending")
    for i in range(100):
        DM.add_data_array(data=[1, 2, 3])
        time.sleep(0.1)
    
    DM.close()
    print("Done")
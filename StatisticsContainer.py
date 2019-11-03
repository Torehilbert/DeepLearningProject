import threading


class Data:
    def __init__(self, maxsize=10000):
        self.lock = threading.Lock()
        self.data = []
    
    def add_data(self, data):
        self.lock.acquire()
        self.data.append(data)
        self.lock.release()

    def extract_data(self):
        self.lock.acquire()
        dataCopy = self.data.copy()
        self.data = []
        self.lock.release()
        return dataCopy

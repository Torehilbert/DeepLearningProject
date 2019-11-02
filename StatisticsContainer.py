# Stream-ability should be implemented for this class, 
# currently it saves ALL data


class MyData:
    def __init__(self, count, maxsizes=100):
        self.n = count
        if(hasattr(maxsizes, '__len__')):
            if(len(maxsizes) != count):
                msg = "maxsizes need to be scalar or of length 'count'"
                raise Exception(msg)
        else:
            maxsizes = [maxsizes]*count

        self.sizes = []
        self.data = []
        self.cursor = []
        for i in range(count):
            size = maxsizes[i]
            self.sizes.append(size)
            self.cursor.append(0)
            self.data.append([None]*size)

    def add_data(self, data, id):
        if(self.cursor[id] == self.sizes[id]):
            return
        dList = self.data[id]
        dList[self.cursor[id]] = data
        self.cursor[id] += 1

    def extract_recent(self, id, count):
        cursor = self.cursor[id]
        if(cursor >= count):
            return self.data[id][cursor-count:cursor]
        else:
            raise Exception("Not enough data!")

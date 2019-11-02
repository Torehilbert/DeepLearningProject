import time
import numpy as np


class TimeManager:
    def __init__(self, number_of_timers):
        self.number_of_timers = number_of_timers
        self.t0 = None
        self.timeTotal = None
        self.bandSeries = []
        self.bandTimers = []
        for i in range(number_of_timers):
            self.bandTimers.append(0)
            self.bandSeries.append([])

    def start(self):
        self.t0 = time.time()
        return self

    def stop(self):
        if(self.t0 is None):
            errMsg = "TimeManager's 'start' needs to be called before 'stop'"
            raise Exception(errMsg)
        self.timeTotal = time.time() - self.t0

        self.meanBands = []
        self.sumBands = []
        self.minBands = []
        self.maxBands = []
        for i in range(len(self.bandSeries)):
            self.sumBands.append(np.sum(self.bandSeries[i]))
            self.meanBands.append(np.mean(self.bandSeries[i]))
            self.minBands.append(np.min(self.bandSeries[i]))
            self.maxBands.append(np.max(self.bandSeries[i]))

        self.fractions = []
        for i in range(len(self.meanBands)):
            self.fractions.append(self.sumBands[i]/self.timeTotal)
        
        seconds = self.timeTotal
        hours = seconds // 3600
        seconds -= hours*3600
        minutes = seconds // 60
        seconds -= minutes*60
        return (hours, minutes, seconds)

    def start_timer(self, id):
        self.bandTimers[id] = time.time()

    def stop_timer(self, id):
        self.bandSeries[id].append(time.time() - self.bandTimers[id])


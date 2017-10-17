
class Chunker:
    def __init__(self, data):
        self.data = data
        self.off = 0

    def batch(self, size):
        low = self.off
        high = (self.off + size) % len(self.data)
        self.off = high
        if high > len(self.data) or low > high: # ROLLOVER!
            return self.data[low:] + self.data[:high]
        else:
            return self.data[low:high]


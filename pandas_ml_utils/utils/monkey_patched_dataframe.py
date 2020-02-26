from pandas import *


def _values(self):
    print("\n\nlalala\n\n")
    return self._unpatched_values


DataFrame._unpatched_values = DataFrame.values
DataFrame.values = property(_values)

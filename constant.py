dir='C:\\Projects\\FuzzyNeuro\\FuzzyNeuro\\20170417\\4'
index=['open','close','high','low','volume','v_ma20','alpha#6','alpha#23','alpha#28','alpha#54','alpha#101']
lag=25


from os.path import exists
from os import mkdir

if not exists(dir):
    mkdir(dir)

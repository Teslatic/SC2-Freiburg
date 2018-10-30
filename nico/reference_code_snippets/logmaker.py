import time
import datetime

buffer_size = 1 # This makes it so changes appear without buffering
with open('output.log', 'a', buffer_size) as f:
    while(True):
        f.write('{}\n'.format(datetime.datetime.now()))
        time.sleep(1)

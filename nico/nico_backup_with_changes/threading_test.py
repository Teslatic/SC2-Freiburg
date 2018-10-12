#!/usr/bin/python
from threading import Thread
import time

# Define a function for the thread
def print_time( threadName, delay):
   count = 0
   while count < 5:
      time.sleep(delay)
      count += 1
      print("%s: %s" % ( threadName, time.ctime(time.time()) ))


def main():
    # Create two threads as follows
    try:
        t1 = Thread(target=print_time, args=("Thread-1", 1))
        t2 = Thread(target=print_time, args=("Thread-2", 2))

        t1.start()
        t2.start()
        print("Main completed")

    except:
        print ("Error: unable to start thread")

if __name__ == '__main__':
    main()

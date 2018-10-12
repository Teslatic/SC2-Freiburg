import time
import datetime


def print_timestamp(string = ""):
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S: ")+string)

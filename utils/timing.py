# Modified from https://stackoverflow.com/questions/1557571/how-do-i-get-time-of-a-python-programs-execution/12344609#12344609
import atexit
from time import time, strftime, localtime
from datetime import timedelta

try:
    from mpi4py import MPI
    mpi_enabled = True
except ImportError:
    mpi_enabled = False

def secondsToStr(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))

def log(s, elapsed=None):
    line = "-"*40
    if mpi_enabled:
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            print(line)
            print(f"{secondsToStr()} - {s}")
            if elapsed:
                print(f"Elapsed time: {elapsed}")
            print(line)
    else:
        print(line)
        print(f"{secondsToStr()} - {s}")
        if elapsed:
            print(f"Elapsed time: {elapsed}")
        print(line)

def endlog():
    end = time()
    elapsed = end - start
    log("End Program", secondsToStr(elapsed))

start = time()
atexit.register(endlog)
log("Start Program")

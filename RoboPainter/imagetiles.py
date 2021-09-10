"""
This module contains functions to split images to easier processing
"""
import threading
import typing

import numpy as np
import math

from tqdm import tqdm

TileLength = 400

from threading import Lock

s_print_lock = Lock()
def s_print(*a, **b):
    """Thread safe print function"""
    with s_print_lock:
        print(*a, **b)

def splitimager(im: np.ndarray, callback: typing.Callable,tileLength = TileLength):

    noRows = math.ceil(im.shape[0] / tileLength)
    noColumns = math.ceil(im.shape[1] / tileLength)
    total_ = TileLength*noRows*noColumns
    bar = tqdm(total=total_)
    print("Starting threads...")
    threads = [threading.Thread(target=callback, args=(im,c * TileLength,r * TileLength,TileLength,bar))
               for r in range(noRows)
               for c in range(noColumns)]

    for thread in threads:
        thread.start()

    s_print("waiting for threads")
    for thread in threads:
        thread.join()
    bar.close()


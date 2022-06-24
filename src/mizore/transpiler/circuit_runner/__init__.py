
import os
if "OMP_NUM_THREADS" not in os.environ.keys():
    raise Exception("Must specify a OMP_NUM_THREADS")

import torch
import os, sys

sys.path.append(os.path.dirname(__file__) + "/../ktransformers_ext/llamafile/build")
print(os.path.dirname(__file__) + "/../ktransformers_ext/llamafile/build")
import cpuinfer_ext

class CPUInfer:
    def __init__(self, thread_num):
        self.cpuinfer = cpuinfer_ext.CPUInfer(thread_num)

    def submit(self, task):
        fn, args = task
        self.cpuinfer.submit(fn, *args)

    def sync(self):
        self.cpuinfer.sync()
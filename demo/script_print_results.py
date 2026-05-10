import sys
sys.path.append('../src')
from CounterfactualLoader import CounterfactualLoader
import numpy as np

def getCounterfactualRadius(loader, sample):
    counterfactual_index = 0
    radius = 0
    factual = loader.getGenerated(sample).astype(np.float64)/255
    while True:
        try:
            counterfactual = loader.getGenerated(sample, counterfactual_index).astype(np.float64)/255
            distance = np.sum((factual - counterfactual)**2) ** 0.5
            radius = max(radius, distance)
        except:
            break
        counterfactual_index += 1
    return radius

loader = CounterfactualLoader("demo.ctf")
for i in range(10):
    print("CR for sample {}: {:.3f}".format(i, getCounterfactualRadius(loader, i)))

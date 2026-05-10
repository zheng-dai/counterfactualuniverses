import sys
sys.path.append('../src')
from DiffusionEnsemble import diffusionEnsemble, createCode

from CounterfactualLoader import CounterfactualLoader

from PIL import Image
import numpy as np

loader = CounterfactualLoader("demo.ctf")
index = int(input("Enter which counterfactual universe you would like to visualize from test.ctf (0-{}):\n".format(loader.GeneratedSetSize-1)))

factual = loader.getGenerated(index)
counterfactual = np.stack([loader.getGenerated(index, i) for i in range(1000)])
training = np.stack([loader.getTraining(i)[0] for i in range(1000)])

factual = factual.reshape(32, 32)
counterfactual = counterfactual.reshape(25, 40, 32, 32).transpose(0,2,1,3).reshape(25*32,-1)
training = training.reshape(25, 40, 32, 32).transpose(0,2,1,3).reshape(25*32,-1)

row1 = np.zeros((32, 32*39), dtype=np.uint8) + 255
row2 = np.zeros((32, 32*40), dtype=np.uint8) + 255
summary = np.concatenate([
    np.concatenate((factual, row1), axis=1),
    row2,
    counterfactual,
    row2,
    training
], axis=0)
im = Image.fromarray(summary)
im.save("./out.png")

print ("Output written to out.png")
# from CounterfactualLoader import CounterfactualLoader
# loader = CounterfactualLoader(<path/to/filename>)
# loader.getGenerated(0)
# loader.getTraining(10)
# loader.getGenerated(0, missing = 10)

import json
import tarfile
import numpy as np
from PIL import Image

class SimpleIterator:
    def __init__(self, dataset):
        self.dataset = dataset
        
    def length(self):
        return self.dataset.shape[0]
    
    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self, i):
        return self.dataset[i:i+1]
    
class LazyIterator:
    def __init__(self, files, loadimage):
        self.files = files
        self.sizes = [int(file[:-4].split('_')[-1]) for file in self.files]
        self.loadimage = loadimage
        
    def length(self):
        return len(files)
    
    def __len__(self):
        return len(files)
    
    def __getitem__(self, i):
        return self.loadimage(self.files[i])[:self.sizes[i]]
    
class CounterfactualLoader:
    def __init__(self, filename):
        self._filename = filename
        self._getConfig()
        self._getTraining()
        
    def _loadImage(self, filename):
        if filename[:5] == "train":
            width, height = self._TrainDim[1], self._TrainDim[0]
        else:
            width, height = self._GenDim[1], self._GenDim[0]
        with tarfile.open(self._filename, 'r') as tar:
            with tar.extractfile(filename) as file:
                image = Image.open(file)
                image = np.array(image)
                dim1 = image.shape[0]//height
                dim2 = image.shape[1]//width
                image = np.array(image)
                image = image.reshape(dim1, height, dim2, width, -1)
                image = image.transpose(0,2,1,3,4)
                image = image.reshape(dim1*dim2, height, width, -1)
        return image
    
    def _getConfig(self):
        with tarfile.open(self._filename, 'r') as tar:
            names = [member.name for member in tar.getmembers()]
            names = list(filter(lambda x: x[:5] == "leave", names))
            names.sort(key = lambda x : int(x.split('_')[1]))
            with tar.extractfile("config.json") as file:
                config = json.load(file)
        self._Format = config['Format']
        self.TrainingSetSize = config['TrainingUnits']
        self.GeneratedSetSize = config['NumberGenerated']
        self._TrainDim = [
            config['TrainHeight'],
            config['TrainWidth'],
            config['TrainChannel']
        ]
        self._GenDim = [
            config['GenHeight'],
            config['GenWidth'],
            config['GenChannel']
        ]
        self._Counterfactual = names
        self._Factual = self._loadImage("factual.png")
        
    def _getTraining(self):
        if self._Format == "ONEFILE":
            self._training = SimpleIterator(self._loadImage("training_set.png"))
        else:
            with tarfile.open(self._filename, 'r') as tar:
                names = [member.name for member in tar.getmembers()]
                names = list(filter(lambda x: x[:5] == "train", names))
                names.sort(key = lambda x : int(x.split('_')[1]))
            self._training = LazyIterator(names, self._loadImage)
        
    # Returns the ith generated image
    # If missing specified, returns the counterfactual where the training set is incomplete
    def getGenerated(self, i, missing = None):
        if missing == None:
            return self._Factual[i]
        else:
            return self._loadImage(self._Counterfactual[missing])[i]
        
    # Returns a list of images that belong to unit i
    def getTraining(self, i):
        return self._training[i]

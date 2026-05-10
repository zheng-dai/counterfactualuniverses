import sys
sys.path.append('../src')
from DiffusionEnsemble import diffusionEnsemble, createCode

from PIL import Image
import torchvision
import torch
import numpy as np
import tarfile
import os
import uuid

# Change this to control how many counterfactual universes to generate
batchsize = 10
# Change this to change the name of the file to save the universe to
output_file = "demo.ctf"
# Temporary working directory
working_directory = ".temp-" + str(uuid.uuid4())

# Utility functions
def toImage(tensor):
    arr = ((tensor/2.) + 0.5).clip(0, 1).numpy()
    arr = np.uint8(255 * arr).squeeze().reshape(-1, 32)
    return Image.fromarray(arr)

def saveImage(tensor, fname):
    image = toImage(tensor)
    image.save(fname)

    
code = createCode(22, 1000, 12345)
MNIST_dataset = torchvision.datasets.MNIST(
    root = "/data/gl/g4/zhengdai/2024/attribution/data",
    train = False,
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.CenterCrop((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ]
    ),
    download = True
)

ensemble = diffusionEnsemble(code, MNIST_dataset, 1, "demo")
ensemble.loadEnsemble(verbose=True)

# Hardware acceleration
ensemble.cuda()

# Initialize file
os.makedirs(working_directory)
with open(working_directory+"/config.json", 'wt') as fout:
    fout.write(f"""{{
    "Format": "ONEFILE",
    "TrainingUnits": 1000,
    "NumberGenerated": {batchsize},
    "TrainHeight": 32,
    "TrainWidth": 32,
    "TrainChannel": 1,
    "GenHeight": 32,
    "GenWidth": 32,
    "GenChannel": 1
}}""")    
training_set = torch.stack([MNIST_dataset[i][0] for i in range(1000)])
saveImage(training_set, f"{working_directory}/training_set.png")

# Main universe generation loop
noise = ensemble.generateNoiseTrajectory(batchsize=batchsize)
factual = ensemble.sample(noise)
saveImage(factual, f"{working_directory}/factual.png")
for i in range(1000):
    counterfactual = ensemble.sample(noise, code=i)
    saveImage(counterfactual, f"{working_directory}/leave_{i+1}_out.png")
    
# Create counterfactual file
with tarfile.open(output_file, "w") as tar:
    for fn in os.listdir(working_directory):
        p = os.path.join(working_directory, fn)
        tar.add(p, arcname=fn)
        
# Cleanup
for file in os.listdir(working_directory):
    os.remove(f"{working_directory}/{file}")
os.rmdir(working_directory)

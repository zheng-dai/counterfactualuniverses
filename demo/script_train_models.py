import sys
sys.path.append('../src')
from DiffusionEnsemble import diffusionEnsemble, createCode
import torchvision

DATASET_SIZE = 1000
code = createCode(22, DATASET_SIZE, 12345)
MNIST_dataset = torchvision.datasets.MNIST(
    root = "./MNIST",
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
for ensemble_member in range(22):
    print ("Training model {}/{}".format(ensemble_member+1, code.shape[1]))
    ensemble.train(ensemble_member, num_epochs=1000000//DATASET_SIZE)
    ensemble.clean()

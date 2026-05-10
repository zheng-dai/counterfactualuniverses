from diffusers import UNet2DModel
from diffusers import DDPMScheduler
from accelerate import Accelerator
from tqdm import tqdm

import torch
import pickle
import numpy as np

def createCode(n, N, seed=None):
    """
    Function for creating a random code for use in diffusion ensemble.

    Arguments:
        n (int): Number of models in ensemble
        N (int): Number of training samples
        seed (int): Optional -- create the code deterministically from the seed, otherwise the code will be random.
    """
    if seed is None:
        rng = np.random
    else:
        rng = np.random.default_rng(seed)
    S = set()
    base = np.concatenate((np.ones(n//2), np.zeros(n//2)))
    while len(S) < N:
        code = tuple(base[rng.permutation(n)])
        S.add(code)
    S = np.stack([np.array(code) for code in S])
    return S

def _sampleStep(t, prev_t, sample, model_output, noise):
    num_train_timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    alpha_prod_t = alphas_cumprod[t]
    alpha_prod_t_prev = alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t
    
    # Get prediction
    b1 = (beta_prod_t**0.5)
    a1 = (alpha_prod_t**0.5)
    pred_original_sample = (sample - (b1 * model_output)) / a1
    
    # Interpolate
    pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
    pred_prev_sample = (pred_original_sample_coeff * pred_original_sample) + (current_sample_coeff * sample)
    
    # Add noise
    variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
    variance = torch.clamp(variance, min=1e-20)
    noise = (variance ** 0.5) * noise
    return pred_prev_sample + noise

def _getSchedule(num_inference_steps):
    step_ratio = 1000 // num_inference_steps
    timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].astype(np.int64)
    return timesteps, timesteps - step_ratio

@torch.no_grad()
def _inference(model, noise_array, code=None):
    device = next(model.parameters()).device
    num_inference_steps = noise_array.shape[0] - 1
    noisetensor = torch.tensor(noise_array, device = device, dtype = torch.float32)
    image = noisetensor[0].to(device)
    
    for t, prev_t, noise in tqdm(
        list(zip(*_getSchedule(num_inference_steps), noisetensor[1:])),
        position=0,
        leave=True
    ):
        out = model.forward(image, t, code)
        image = _sampleStep(t, prev_t, image, out, noise)
    image = image.cpu()
    return image

class diffusionEnsembleModule(torch.nn.Module):
    def __init__(self, models):
        super(diffusionEnsembleModule, self).__init__()
        self.config = models[0].config
        self.models = torch.nn.ModuleList(models)
        self.ensemble_size = len(models)
        
    def forward(self, sample, timestep, code=None):
        if code is None:
            x = torch.stack([
                model(sample, timestep, return_dict=False)[0]
                for model in self.models
            ])
        else:
            # Keep the models that haven't seen this unit of data
            x = torch.stack([
                model(sample, timestep, return_dict=False)[0]
                for include, model in zip(code, self.models) if include == 0
            ])
        return torch.mean(x, dim = 0)

class diffusionEnsemble:
    """
    A diffusion ensemble class that can be used to train, load, and sample diffusion ensembles.

    Functions:
        train: Train a model within the ensemble
        loadEnsemble: If all ensemble members are trained, load the whole ensemble
        sample: Generate a sample using the ensemble if loaded
    """
    
    def __init__(self, code, dataset, channels, label):
        """
        Initialize a diffusion ensemble

        Arguments:
            code (np.array): a Nxn numpy array of 0s and 1s. N is the training set size, while n is the number of models in the ensemble
            dataset (torch.utils.data.Dataset): dataset to train on as image, label pairs. It must contain at least N entries. If dataset contains more than N entries, only the first N samples are used for training
            channels (int): the number of channels of images. This should be 1 for monochrome images and 3 for RGB. For embeddings this is generally 4
            label (str): the relative directory to save to or retrieve the model from
        """
        self.dataset = dataset
        self.code = code
        self.channels = channels
        self.identifier = label
        self.model = None
        
    def generateNoiseTrajectory(self, batchsize=1):
        return np.random.normal(0, 1, (51, batchsize, self.channels, 32, 32))
        
    def sample(self, noise=None, code=None):
        """
        Sample a diffusion ensemble

        Arguments:
            noise (np.array): the noise used to generate the trajectory. Random if None. Use generateNoiseTrajectory to generate noise that can be reused
            code (int or np.array): if int, will use the ith entry in the code used to initialize the model. Any entry in the code that is not zero will have its corresponding model removed during sampling
        """
        noise = self.generateNoiseTrajectory() if noise is None else noise
        if isinstance(code, int):
            code = self.code[code]
        return _inference(self.model, noise, code)
        
    def getTrainingSetForShard(self, shard):
        index = np.argwhere(self.code[:, shard] > 0).reshape(-1)
        return torch.utils.data.Subset(self.dataset, index)
    
    def getFreshModel(self, channels, hiddenUnits=(128, 256, 512, 512)):
        model = UNet2DModel(
            sample_size=32,
            in_channels=channels,
            out_channels=channels,
            layers_per_block=2,
            block_out_channels=hiddenUnits,
            down_block_types=( 
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ), 
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D"  
            ),
        )
        return model
    
    def cuda(self):
        return self.model.cuda()
    
    def cpu(self):
        return self.model.cpu()
    
    def loadEnsemble(self, root="./", verbose=False):
        path = root + self.identifier + "/Shard{}/pytorch_model.bin"
        models = []
        for i in range(self.code.shape[1]):
            fpath = path.format(i)
            if verbose:
                print ("Loading model from {}...".format(fpath))
            model = self.getFreshModel(self.channels)
            ckpt = torch.load(path.format(i), map_location = "cpu")
            model.load_state_dict(ckpt)
            model.eval()
            models.append(model)
        self.model = diffusionEnsembleModule(models)
    
    def train(self, shard, num_epochs, lr=1e-4):
        """
        Train one member of the diffusion ensemble.

        Arguments:
            shard (int): Which model to train. Ranges from 0 to n-1 inclusive
            num_epochs (int): Number of epochs to train for
        """
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        model = self.getFreshModel(self.channels).cuda()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_dataset = self.getTrainingSetForShard(shard)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=min(128, len(train_dataset)),
            shuffle=True,
            drop_last = True
        )
        
        accelerator = Accelerator(
            device_placement = False,
            mixed_precision = "fp16",
            gradient_accumulation_steps = 1
        )

        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )
        model.train()

        trace = []
        for epoch in tqdm(range(num_epochs), position = 0, leave = True):
            with tqdm(train_dataloader, position = 1, leave = False) as pbar:
                for images, labels in pbar:
                    clean_images = images.cuda()
                    noise = torch.randn(clean_images.shape).to(clean_images.device)
                    batchsize = clean_images.shape[0]
                    
                    timesteps = torch.randint(0, 1000, (batchsize,), device=clean_images.device).long()
                    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                    
                    with accelerator.accumulate(model):
                        noise_pred = model(noisy_images,
                                           timesteps,
                                           return_dict=False)[0]
                        loss = torch.nn.functional.mse_loss(noise_pred, noise)
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()

                        trace.append(loss.item())
                    pbar.set_description("{:.5f}".format(trace[-1]))

        accelerator.save_model(model, "./{}/Shard{}".format(self.identifier, shard), safe_serialization = False)
        with open("./{}/Shard{}/trace.pkl".format(self.identifier, shard), 'wb') as fout:
            pickle.dump(trace, fout)
        accelerator.free_memory()
            
    def clean(self):
        torch.cuda.empty_cache()

from typing import List

import torch
import torch.utils.data
import torchvision
from PIL import Image

from labml import lab, tracker, experiment, monit
from labml.configs import BaseConfigs, option
from labml_helpers.device import DeviceConfigs
from labml_nn.diffusion.ddpm import DenoiseDiffusion
from labml_nn.diffusion.ddpm.unet import UNet
from fedavg import aggregate,dirichlet_loaders
import numpy as np

class Configs(BaseConfigs):
    """
    ## Configurations
    """

    device: torch.device = DeviceConfigs()

    # federated init
    models=[]
    diffusions=[]
    optimizers=[]
    data_size=[]
    dataloaders=[]
    global_model: UNet
    global_diffusion: DenoiseDiffusion
    # Default settings
    # Number of channels in the image. $3$ for RGB.
    image_channels: int = 3
    # Image size
    image_size: int = 32
    # Number of channels in the initial feature map
    n_channels: int = 64
    # The list of channel numbers at each resolution.
    # The number of channels is `channel_multipliers[i] * n_channels`
    channel_multipliers: List[int] = [1, 2, 2, 4]
    # The list of booleans that indicate whether to use attention at each resolution
    is_attention: List[int] = [False, False, False, True]

    # Number of time steps $T$
    n_steps: int = 1_000
    # Batch size
    batch_size: int = 64
    # Number of samples to generate
    n_samples: int = 16
    # Learning rate
    learning_rate: float = 2e-5
    # Number of federated clients
    n_clients: int = 3
    # Number of local training epochs
    local_iters: int = 1
    # Number of training epochs
    epochs: int = 1_000

    # Dataset
    dataset: torch.utils.data.Dataset
    # Dataloader
    # data_loader: torch.utils.data.DataLoader

    # Adam optimizer
    # optimizer: torch.optim.Adam

    def init(self):
        # Create $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model


        # total_size = 0
        self.dataloaders=dirichlet_loaders(self.dataset,self.n_clients,batch_size=self.batch_size)
        self.global_model=UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        ).to(self.device)
        self.global_diffusion=DenoiseDiffusion(
            eps_model=self.global_model,
            n_steps=self.n_steps,
            device=self.device,
            )
        for i in range(self.n_clients):
            size = len(self.dataloaders[i])
            # total_size += size
            self.data_size.append(size)
            model=UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
            ).to(self.device)
            self.models.append(model)
            diffusion=DenoiseDiffusion(
            eps_model=self.models[i],
            n_steps=self.n_steps,
            device=self.device,
            )
            self.diffusions.append(diffusion)
            optimizer = torch.optim.Adam(self.models[i].parameters(), lr=self.learning_rate)
            self.optimizers.append(optimizer)

        # self.data_size = np.array(self.data_size) / total_size
        # Image logging
        tracker.set_image("sample", True)

    def sample(self):
        """
        ### Sample images
        """
        with torch.no_grad():
            # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
            x = torch.randn([self.n_samples, self.image_channels, self.image_size, self.image_size],
                            device=self.device)

            # Remove noise for $T$ steps
            for t_ in monit.iterate('Sample', self.n_steps):
                # $t$
                t = self.n_steps - t_ - 1
                # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
                x = self.global_diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

            # Log samples
            tracker.save('sample', x)

    def train(self,idx):
        """
        ### Train
        """

        # Iterate through the dataset
        for data in monit.iterate('Train', self.dataloaders[idx]):
            # Increment global step
            tracker.add_global_step()
            # Move data to device
            data = data.to(self.device)

            # Make the gradients zero
            self.optimizers[idx].zero_grad()
            # Calculate loss
            loss = self.diffusions[idx].loss(data)
            # Compute gradients
            loss.backward()
            # Take an optimization step
            self.optimizers[idx].step()
            # Track the loss
            tracker.save('loss', loss)

    def run(self):
        """
        ### Training loop
        """
        for _ in monit.loop(self.epochs):
            # Train the model
            print("Epoch ", _+1)
            for i in monit.loop(self.n_clients):
                print("Client: ", i+1)
                # Download global model
                self.models[i].load_state_dict(self.global_model.state_dict())
                for j in range(self.local_iters):
                    # Local training
                    self.train(i)
                    tracker.new_line()
            # Model aggregation
            aggregated_dict = aggregate(self.global_model, self.models, self.data_size)
            print("Model aggregation")
            self.global_model.load_state_dict(aggregated_dict)
            # Sample some images of global diffusion
            self.sample()
            # New line in the console
            tracker.new_line()
            # Save the model
            experiment.save_checkpoint()

    def summary(self):
        print (self.global_model)


class CelebADataset(torch.utils.data.Dataset):
    """
    ### CelebA HQ dataset
    """

    def __init__(self, image_size: int):
        super().__init__()

        # CelebA images folder
        folder = lab.get_data_path() / 'celebA'
        # List of files
        self._files = [p for p in folder.glob(f'**/*.jpg')]

        # Transformations to resize the image and convert to tensor
        self._transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        """
        Size of the dataset
        """
        return len(self._files)

    def __getitem__(self, index: int):
        """
        Get an image
        """
        img = Image.open(self._files[index])
        return self._transform(img)


@option(Configs.dataset, 'CelebA')
def celeb_dataset(c: Configs):
    """
    Create CelebA dataset
    """
    return CelebADataset(c.image_size)


class MNISTDataset(torchvision.datasets.MNIST):
    """
    ### MNIST dataset
    """

    def __init__(self, image_size):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

        super().__init__(str(lab.get_data_path()), train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)[0]


@option(Configs.dataset, 'MNIST')
def mnist_dataset(c: Configs):
    """
    Create MNIST dataset
    """
    return MNISTDataset(c.image_size)


def main():
    # Create experiment

    experiment.create(name='diffuse', writers={'screen', 'labml'})

    # Create configurations

    configs=Configs()
    # Set configurations. You can override the defaults by passing the values in the dictionary.
    experiment.configs(configs, {
        'dataset': 'MNIST',  # 'MNIST','CelebA'
        'image_channels': 1,  # 1,3
        'epochs': 5,  # 5,100
        'n_clients': 5,
    })



    configs.init()

    experiment.add_pytorch_models({'eps_model': configs.global_model})

    with experiment.start():
        configs.run()

    #configs.summary()
#
if __name__ == '__main__':
    main()
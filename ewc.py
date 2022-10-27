import itertools
import warnings
from typing import List

import torch
import torch.utils.data
import torchvision
from PIL import Image
from collections import defaultdict

from avalanche.training.utils import zerolike_params_dict, copy_params_dict
from labml import lab, tracker, experiment, monit
from labml.configs import BaseConfigs, option
from labml_helpers.device import DeviceConfigs
from labml_nn.diffusion.ddpm import DenoiseDiffusion
from labml_nn.diffusion.ddpm.unet import UNet
from torch.utils.data import DataLoader


class Configs(BaseConfigs):
    """
    ## Configurations
    """
    # Device to train the model on.
    # [`DeviceConfigs`](https://docs.labml.ai/api/helpers.html#labml_helpers.device.DeviceConfigs)
    #  picks up an available CUDA device or defaults to CPU.
    device: torch.device = DeviceConfigs()

    # U-Net model for $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
    eps_model: UNet
    # [DDPM algorithm](index.html)
    diffusion: DenoiseDiffusion

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

    # Number of training epochs
    epochs: int = 1_000

    # # Dataset
    # dataset: torch.utils.data.Dataset
    # # Dataloader
    # data_loader: torch.utils.data.DataLoader

    # Adam optimizer
    optimizer: torch.optim.Adam
    # ewc settings
    ewc_lambda: int = 10000
    mode: str = "separate"
    decay_factor = None
    keep_importance_data = False
    saved_params: defaultdict
    importances: defaultdict
    dataloaders: []
    datasets: []

    # todo: implement multitask

    def init(self):
        # Create $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        ).to(self.device)

        # Create [DDPM class](index.html)
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
        )
        self.data_loader()
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.learning_rate)

        # Image logging
        tracker.set_image("sample", True)

        if self.mode == "separate":
            self.keep_importance_data = True

        self.saved_params = defaultdict(list)
        self.importances = defaultdict(list)

    def data_loader(self):
        dataloaders = []
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.image_size),
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.SVHN('./data/SVHN', split='train', download=True, transform=transform)

        # Create dataloader
        data_loader = torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=True, pin_memory=True)
        dataloaders.append(data_loader)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.image_size),
            torchvision.transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.MNIST('./data/MNIST', train=True, download=True, transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=True, pin_memory=True)
        dataloaders.append(data_loader)
        self.dataloaders = dataloaders

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
                x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

            # Log samples
            tracker.save('sample', x)

    def train(self, i):
        """
        ### Train
        """

        # Iterate through the dataset
        # todo: implement multitask
        for data in monit.iterate('Train', self.dataloaders[i]):
            # Increment global step
            tracker.add_global_step()
            # Move data to device
            data = data[0].to(self.device)

            # Make the gradients zero
            self.optimizer.zero_grad()
            # Calculate loss
            loss = self.diffusion.loss(data)
            penalty = self.penalty(i)
            loss += self.ewc_lambda * penalty
            # Compute gradients
            loss.backward()
            # Take an optimization step
            self.optimizer.step()
            # Track the loss
            tracker.save('loss', loss)

    def run(self):
        """
        ### Training loop
        """

        for i in range(len(self.dataloaders)):
            task_counter = 0
            for _ in monit.loop(self.epochs):
                # Train the model
                self.train(i)
                # Sample some images
                self.sample()
                # New line in the console
                tracker.new_line()
                # Save the model
                experiment.save_checkpoint()
                # after_training_exp
            exp_counter = task_counter
            importances = self.compute_importances(
                self.eps_model,
                self.optimizer,
                self.dataloaders[i],
                self.device,
                self.batch_size,
            )
            self.update_importances(importances, exp_counter)
            self.saved_params[exp_counter] = copy_params_dict(self.eps_model)
            # clear previous parameter values
            if exp_counter > 0 and (not self.keep_importance_data):
                del self.saved_params[exp_counter - 1]
            task_counter += 1

    def penalty(self, task_counter):
        penalty = torch.tensor(0).float().to(self.device)

        if self.mode == "separate":
            for experience in range(task_counter):
                for (_, cur_param), (_, saved_param), (_, imp) in zip(
                        self.eps_model.named_parameters(),
                        self.saved_params[experience],
                        self.importances[experience],
                ):
                    # dynamic models may add new units
                    # new units are ignored by the regularization
                    n_units = saved_param.shape[0]
                    cur_param = cur_param[:n_units]
                    penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        elif self.mode == "online":
            # todo: task_counter
            prev_exp = task_counter - 1
            for (_, cur_param), (_, saved_param), (_, imp) in zip(
                    self.eps_model.named_parameters(),
                    self.saved_params[prev_exp],
                    self.importances[prev_exp],
            ):
                # todo: dynamic models may add new units
                # new units are ignored by the regularization
                n_units = saved_param.shape[0]
                cur_param = cur_param[:n_units]
                penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        else:
            raise ValueError("Wrong EWC mode.")

        return penalty

    def compute_importances(
            self, model, optimizer, dataloader, device, batch_size
    ):
        """
        Compute EWC importance matrix for each parameter
        """

        model.eval()

        # Set RNN-like modules on GPU to training mode to avoid CUDA error
        if device == "cuda":
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        "RNN-like modules do not support "
                        "backward calls while in `eval` mode on CUDA "
                        "devices. Setting all `RNNBase` modules to "
                        "`train` mode. May produce inconsistent "
                        "output if such modules have `dropout` > 0."
                    )
                    module.train()

        # list of list
        importances = zerolike_params_dict(model)
        # collate_fn = (
        #     dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        # )
        # dataloader = DataLoader(
        #     dataset, batch_size=batch_size, collate_fn=collate_fn
        # )
        for i, batch in enumerate(dataloader):
            # get only input, target and task_id from the batch
            # x, y, task_labels = batch[0], batch[1], batch[-1]
            # x, y = x.to(device), y.to(device)
            batch = batch[0].to(self.device)
            optimizer.zero_grad()
            # out = avalanche_forward(model, x, task_labels)
            loss = self.diffusion.loss(batch)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                    model.named_parameters(), importances
            ):
                assert k1 == k2
                if p.grad is not None:
                    imp += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances:
            imp /= float(len(dataloader))

        return importances

    @torch.no_grad()
    def update_importances(self, importances, t):
        """
        Update importance for each parameter based on the currently computed
        importances.
        """

        if self.mode == "separate" or t == 0:
            self.importances[t] = importances
        elif self.mode == "online":
            for (k1, old_imp), (k2, curr_imp) in itertools.zip_longest(
                    self.importances[t - 1],
                    importances,
                    fillvalue=(None, None),
            ):
                # Add new module importances to the importances value (New head)
                if k1 is None:
                    self.importances[t].append((k2, curr_imp))
                    continue

                assert k1 == k2, "Error in importance computation."

                self.importances[t].append(
                    (k1, (self.decay_factor * old_imp + curr_imp))
                )

            # clear previous parameter importances
            if t > 0 and (not self.keep_importance_data):
                del self.importances[t - 1]

        else:
            raise ValueError("Wrong EWC mode.")


def main():
    # Create experiment
    experiment.create(name='diffuse', writers={'screen', 'labml'})

    # Create configurations
    configs = Configs()

    # Set configurations. You can override the defaults by passing the values in the dictionary.
    experiment.configs(configs, {
        # 'dataset': 'MNIST',  # 'MNIST'，CelebA
        'image_channels': 1,  # 1,3
        'epochs': 50,  # 5,100
    })

    # Initialize
    configs.init()

    # Set models for saving and loading
    experiment.add_pytorch_models({'eps_model': configs.eps_model})

    # Start and run the training loop
    with experiment.start():
        configs.run()


#
if __name__ == '__main__':
    main()
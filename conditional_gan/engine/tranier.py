# Copyright (c) AlphaBetter. All rights reserved.
import time
from copy import deepcopy
from pathlib import Path

import torch
import torchvision.datasets
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch import optim
from torch.nn import functional as F_torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision import utils as vutils

from conditional_gan.models import *
from conditional_gan.utils.checkpoint import load_state_dict, save_checkpoint, strip_optimizer
from conditional_gan.utils.envs import select_device, set_seed_everything
from conditional_gan.utils.events import LOGGER, AverageMeter, ProgressMeter
from conditional_gan.utils.ops import increment_name
from conditional_gan.utils.torch_utils import get_model_info

__all__ = [
    "Trainer", "init_train_env",
]


class Trainer:
    def __init__(self, config_dict: DictConfig, device: torch.device) -> None:
        self.config_dict = config_dict
        self.device = device

        self.exp_name = self.config_dict.EXP_NAME
        self.dataset_config_dict = self.config_dict.DATASET
        self.model_config_dict = self.config_dict.MODEL
        self.train_config_dict = self.config_dict.TRAIN

        # ========== Init all config ==========
        # datasets
        self.dataset_train_name = self.dataset_config_dict.TRAIN.NAME
        self.dataset_train_root = self.dataset_config_dict.TRAIN.ROOT
        self.dataset_train_image_size = self.dataset_config_dict.TRAIN.IMAGE_SIZE
        self.dataset_train_normalize_mean = list(self.dataset_config_dict.TRAIN.NORMALIZE.MEAN)
        self.dataset_train_normalize_std = list(self.dataset_config_dict.TRAIN.NORMALIZE.STD)

        # train
        # train weights
        self.g_weights_path = self.train_config_dict.get("G_WEIGHTS_PATH", "")
        self.d_weights_path = self.train_config_dict.get("D_WEIGHTS_PATH", "")
        # train dataset
        self.train_image_size = self.train_config_dict.get("IMAGE_SIZE", 28)
        self.train_batch_size = self.train_config_dict.get("BATCH_SIZE", 128)
        self.train_num_workers = self.train_config_dict.get("NUM_WORKERS", 4)
        # train loss
        self.adv_loss_function = self.train_config_dict.LOSS.get("ADV", None)
        if self.adv_loss_function:
            self.adv_loss_type = self.adv_loss_function.get("TYPE", "mse_loss")
            self.adv_loss_weight = self.adv_loss_function.get("WEIGHT", 1.0)
        # train hyper-parameters
        self.epochs = self.train_config_dict.EPOCHS
        # train setup
        self.local_rank = self.train_config_dict.LOCAL_RANK
        self.rank = self.train_config_dict.RANK
        self.world_size = self.train_config_dict.WORLD_SIZE
        self.dist_url = self.train_config_dict.DIST_URL
        self.save_dir = Path(self.train_config_dict.SAVE_DIR)
        # train results
        self.output_dir = Path(self.train_config_dict.OUTPUT_DIR)
        self.verbose = self.train_config_dict.VERBOSE

        # ========== Init all objects ==========
        # datasets
        self.train_dataloader = self.get_dataloader()
        self.num_train_batch = len(self.train_dataloader)

        # model
        self.g_model = self.get_g_model()
        self.d_model = self.get_d_model()

        # optim
        self.g_optimizer = self.get_g_optimizer()
        self.d_optimizer = self.get_d_optimizer()

        # lr_scheduler
        self.g_lr_scheduler = self.get_g_lr_scheduler()
        self.d_lr_scheduler = self.get_d_lr_scheduler()

        # losses
        self.adv_criterion = self.get_loss(self.adv_loss_type)

        self.start_epoch = 0
        self.current_epoch = 0
        # resume model for training
        self.resume()

        # tensorboard
        self.tblogger = SummaryWriter(str(self.save_dir))

        # training variables
        self.start_time: float = 0.0
        self.batch_time: AverageMeter = AverageMeter("Time", ":6.3f")
        self.data_time: AverageMeter = AverageMeter("Data", ":6.3f")
        self.d_losses: AverageMeter = AverageMeter("D loss", ":.4e")
        self.g_losses: AverageMeter = AverageMeter("G loss", ":.4e")
        self.d_x_losses: AverageMeter = AverageMeter("D(x)", ":.4e")
        self.d_g_z1_losses: AverageMeter = AverageMeter("D(G(z1))", ":.4e")
        self.d_g_z2_losses: AverageMeter = AverageMeter("D(G(z2))", ":.4e")

        # eval for training
        if self.model_config_dict.G.TYPE == "vanilla_net":
            self.fixed_noise = torch.randn([self.train_batch_size, self.model_config_dict.G.LATENT_DIM], device=device)
            self.fixed_conditional = torch.randint(0, self.model_config_dict.G.NUM_CLASSES - 1, (self.train_batch_size,), device=device)
        elif self.model_config_dict.G.TYPE == "conv_net":
            self.fixed_noise = torch.randn([self.train_batch_size, self.model_config_dict.G.LATENT_DIM], device=device)
            self.fixed_conditional = torch.randint(0,
                                                   self.model_config_dict.G.NUM_CLASSES - 1,
                                                   (self.train_batch_size, self.model_config_dict.G.NUM_CLASSES),
                                                   device=self.device)
        else:
            raise NotImplementedError(f"Model type `{self.model_config_dict.G.TYPE}` is not implemented.")
        self.save_visual_dir = self.save_dir.joinpath("visual")
        self.save_visual_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the mixed precision method
        self.scaler = torch.amp.GradScaler(enabled=self.device.type != "cpu")

        # Define the path to save the model
        self.save_checkpoint_dir = self.save_dir.joinpath("weights")

        LOGGER.info(f"Results save to `{self.save_dir}`")
        if self.verbose:
            g_model_info = get_model_info(self.g_model, self.train_image_size, self.device)
            d_model_info = get_model_info(self.d_model, self.train_image_size, self.device)
            LOGGER.info(f"G model: {self.g_model}")
            LOGGER.info(f"D model: {self.d_model}")
            LOGGER.info(f"G model summary: {g_model_info}")
            LOGGER.info(f"D model summary: {d_model_info}")

    def get_dataloader(self) -> torch.utils.data.DataLoader:
        if self.dataset_train_name == "mnist":
            train_dataset = torchvision.datasets.MNIST(
                self.dataset_train_root,
                True,
                transform=transforms.Compose([
                    transforms.Resize((self.dataset_train_image_size, self.dataset_train_image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.dataset_train_normalize_mean, self.dataset_train_normalize_std),
                ]),
                download=True,
            )
        elif self.dataset_train_name == "fashion_mnist":
            train_dataset = torchvision.datasets.FashionMNIST(
                self.dataset_train_root,
                True,
                transform=transforms.Compose([
                    transforms.Resize((self.train_image_size, self.train_image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))]),
                download=True,
            )
        elif self.dataset_train_name == "image_folder":
            train_dataset = torchvision.datasets.ImageFolder(
                self.dataset_train_root,
                transform=transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((self.dataset_train_image_size, self.dataset_train_image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.dataset_train_normalize_mean, self.dataset_train_normalize_std),
                ]),
            )
        else:
            raise NotImplementedError(f"Dataset `{self.dataset_train_name}` is not implemented. Only support [`mnist`, `fashion_mnist`].")

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.train_num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

        return train_dataloader

    def get_g_model(self) -> nn.Module:
        model_g_type = self.model_config_dict.G.TYPE
        if model_g_type == "vanilla_net":
            g_model = VanillaNet(image_size=self.model_config_dict.G.IMAGE_SIZE,
                                 channels=self.model_config_dict.G.CHANNELS,
                                 num_classes=self.model_config_dict.G.NUM_CLASSES,
                                 latent_dim=self.model_config_dict.G.LATENT_DIM)
        elif model_g_type == "conv_net":
            g_model = ConvNet(image_size=self.model_config_dict.G.IMAGE_SIZE,
                              channels=self.model_config_dict.G.CHANNELS,
                              num_classes=self.model_config_dict.G.NUM_CLASSES,
                              latent_dim=self.model_config_dict.G.LATENT_DIM)
        else:
            raise NotImplementedError(f"Model type `{model_g_type}` is not implemented.")

        g_model = g_model.to(self.device)
        g_model = torch.compile(g_model)

        if self.g_weights_path:
            LOGGER.info(f"Loading state_dict from {self.g_weights_path} for fine-tuning...")
            g_model = load_state_dict(self.g_weights_path, g_model, device=self.device)

        return g_model

    def get_d_model(self) -> nn.Module:
        model_d_type = self.model_config_dict.D.TYPE
        if model_d_type == "discriminator_for_vanilla":
            d_model = DiscriminatorForVanilla(image_size=self.model_config_dict.G.IMAGE_SIZE,
                                              channels=self.model_config_dict.G.CHANNELS,
                                              num_classes=self.model_config_dict.G.NUM_CLASSES)
        elif model_d_type == "discriminator_for_conv":
            d_model = DiscriminatorForConv(image_size=self.model_config_dict.G.IMAGE_SIZE,
                                           channels=self.model_config_dict.G.CHANNELS,
                                           num_classes=self.model_config_dict.G.NUM_CLASSES)
        else:
            raise NotImplementedError(f"Model type `{model_d_type}` is not implemented.")

        d_model = d_model.to(self.device)
        d_model = torch.compile(d_model)

        if self.d_weights_path:
            LOGGER.info(f"Loading state_dict from {self.d_weights_path} for fine-tuning...")
            d_model = load_state_dict(self.d_weights_path, d_model, device=self.device)

        return d_model

    def get_g_optimizer(self) -> optim:
        optim_type = self.train_config_dict.SOLVER.G.OPTIM.TYPE
        if optim_type not in ["adam"]:
            raise NotImplementedError(f"G optimizer {optim_type} is not implemented. Only support `adam`.")

        g_optimizer = optim.Adam(self.g_model.parameters(),
                                 lr=self.train_config_dict.SOLVER.G.OPTIM.LR,
                                 betas=OmegaConf.to_container(self.train_config_dict.SOLVER.G.OPTIM.BETAS))

        return g_optimizer

    def get_d_optimizer(self) -> optim:
        optim_type = self.train_config_dict.SOLVER.D.OPTIM.TYPE
        if optim_type not in ["adam"]:
            raise NotImplementedError(f"D optimizer {optim_type} is not implemented. Only support `adam`.")

        d_optimizer = optim.Adam(self.d_model.parameters(),
                                 lr=self.train_config_dict.SOLVER.D.OPTIM.LR,
                                 betas=OmegaConf.to_container(self.train_config_dict.SOLVER.D.OPTIM.BETAS))
        return d_optimizer

    def get_g_lr_scheduler(self) -> optim.lr_scheduler:
        lr_scheduler_type = self.train_config_dict.SOLVER.G.LR_SCHEDULER.TYPE
        if lr_scheduler_type not in ["step_lr", "multistep_lr", "constant"]:
            raise NotImplementedError(f"G scheduler {lr_scheduler_type} is not implemented. Only support [`step_lr`, `multistep_lr`, `constant`].")

        if lr_scheduler_type == "step_lr":
            g_lr_scheduler = optim.lr_scheduler.StepLR(
                self.g_optimizer,
                step_size=self.train_config_dict.SOLVER.G.LR_SCHEDULER.STEP_SIZE,
                gamma=self.train_config_dict.SOLVER.G.LR_SCHEDULER.GAMMA)
        elif lr_scheduler_type == "multistep_lr":
            g_lr_scheduler = optim.lr_scheduler.MultiStepLR(
                self.g_optimizer,
                milestones=OmegaConf.to_container(self.train_config_dict.SOLVER.G.LR_SCHEDULER.MILESTONES),
                gamma=self.train_config_dict.SOLVER.G.LR_SCHEDULER.GAMMA)
        else:
            g_lr_scheduler = optim.lr_scheduler.ConstantLR(
                self.g_optimizer,
                factor=self.train_config_dict.SOLVER.G.LR_SCHEDULER.FACTOR,
                total_iters=self.train_config_dict.SOLVER.G.LR_SCHEDULER.TOTAL_ITERS)

        return g_lr_scheduler

    def get_d_lr_scheduler(self) -> optim.lr_scheduler:
        lr_scheduler_type = self.train_config_dict.SOLVER.D.LR_SCHEDULER.TYPE
        if lr_scheduler_type not in ["step_lr", "multistep_lr", "constant"]:
            raise NotImplementedError(f"G scheduler {lr_scheduler_type} is not implemented. Only support [`step_lr`, `multistep_lr`, `constant`].")

        if lr_scheduler_type == "step_lr":
            d_lr_scheduler = optim.lr_scheduler.StepLR(
                self.d_optimizer,
                step_size=self.train_config_dict.SOLVER.D.LR_SCHEDULER.STEP_SIZE,
                gamma=self.train_config_dict.SOLVER.D.LR_SCHEDULER.GAMMA)
        elif lr_scheduler_type == "multistep_lr":
            d_lr_scheduler = optim.lr_scheduler.MultiStepLR(
                self.d_optimizer,
                milestones=OmegaConf.to_container(self.train_config_dict.SOLVER.D.LR_SCHEDULER.MILESTONES),
                gamma=self.train_config_dict.SOLVER.D.LR_SCHEDULER.GAMMA)
        else:
            d_lr_scheduler = optim.lr_scheduler.ConstantLR(
                self.d_optimizer,
                factor=self.train_config_dict.SOLVER.D.LR_SCHEDULER.FACTOR,
                total_iters=self.train_config_dict.SOLVER.D.LR_SCHEDULER.TOTAL_ITERS)

        return d_lr_scheduler

    def get_loss(self, loss_type: str) -> nn:
        if loss_type == "l1_loss":
            criterion = nn.L1Loss()
        elif loss_type == "l2_loss":
            criterion = nn.MSELoss()
        elif loss_type == "bce_loss":
            criterion = nn.BCELoss()
        elif loss_type == "bce_with_logits_loss":
            criterion = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(
                f"Loss type {loss_type} is not implemented. Only support [`l1_loss`, `l2_loss`, `bce_loss`, `bce_with_logits_loss`]."
            )

        criterion = criterion.to(device=self.device)
        return criterion

    def resume(self) -> None:
        resume_g = self.train_config_dict.get("RESUME_G", "")
        if resume_g:
            g_checkpoint = torch.load(resume_g, map_location=self.device)
            if g_checkpoint:
                resume_state_dict = g_checkpoint["model"].float().state_dict()
                self.g_model.load_state_dict(resume_state_dict, strict=True)
                self.start_epoch = g_checkpoint["epoch"] + 1
                self.g_optimizer.load_state_dict(g_checkpoint["optimizer"])
                self.g_lr_scheduler.load_state_dict(g_checkpoint["scheduler"])
                LOGGER.info(f"Resumed g model from epoch {self.start_epoch}")
            else:
                LOGGER.warning(f"Loading state_dict from {resume_g} failed, train from scratch...")

        resume_d = self.train_config_dict.get("RESUME_D", "")
        if resume_d:
            d_checkpoint = torch.load(resume_d, map_location=self.device)
            if d_checkpoint:
                resume_state_dict = d_checkpoint["model"].float().state_dict()
                self.d_model.load_state_dict(resume_state_dict, strict=True)
                self.start_epoch = d_checkpoint["epoch"] + 1
                self.d_optimizer.load_state_dict(d_checkpoint["optimizer"])
                self.d_lr_scheduler.load_state_dict(d_checkpoint["scheduler"])
                LOGGER.info(f"Resumed d model from epoch {self.start_epoch}")
            else:
                LOGGER.warning(f"Loading state_dict from {resume_d} failed, train from scratch...")

    def train(self) -> None:
        try:
            self.before_train_loop()
            for self.current_epoch in range(self.start_epoch, self.epochs):
                self.before_epoch()
                self.train_one_epoch()
                self.after_epoch()

            LOGGER.info(f"Training completed in {(time.time() - self.start_time) / 3600:.3f} hours.")
            g_best_checkpoint_path = Path(self.save_dir).joinpath("weights", "g_best.pkl")
            g_last_checkpoint_path = Path(self.save_dir).joinpath("weights", "g_last.pkl")
            d_best_checkpoint_path = Path(self.save_dir).joinpath("weights", "d_best.pkl")
            d_last_checkpoint_path = Path(self.save_dir).joinpath("weights", "d_last.pkl")
            strip_optimizer(g_best_checkpoint_path)
            strip_optimizer(g_last_checkpoint_path)
            strip_optimizer(d_best_checkpoint_path)
            strip_optimizer(d_last_checkpoint_path)

        except Exception as _:
            LOGGER.error("Training failed.")
            raise
        finally:
            if self.device != "cpu":
                torch.cuda.empty_cache()

    def before_train_loop(self):
        LOGGER.info("Training start...")
        self.start_time = time.time()

    def before_epoch(self):
        self.g_model.train()
        self.d_model.train()
        if self.rank != -1:
            self.train_dataloader.sampler.set_epoch(self.current_epoch)

    def train_one_epoch(self):
        progress = ProgressMeter(
            self.num_train_batch,
            [self.batch_time, self.data_time, self.d_losses, self.g_losses, self.d_x_losses, self.d_g_z1_losses, self.d_g_z2_losses],
            prefix=f"Epoch: [{self.current_epoch}]")

        end = time.time()
        for i, (inputs, target) in enumerate(self.train_dataloader):
            # Move datasets to special device.
            inputs = inputs.to(device=self.device, non_blocking=True)
            target = target.to(device=self.device, non_blocking=True)
            batch_size = inputs.size(0)

            # The real sample label is 1, and the generated sample label is 0.
            real_label = torch.full((batch_size, 1), 1, dtype=inputs.dtype).to(device=self.device, non_blocking=True)
            fake_label = torch.full((batch_size, 1), 0, dtype=inputs.dtype).to(device=self.device, non_blocking=True)
            num_classes = self.model_config_dict.G.NUM_CLASSES
            if self.model_config_dict.G.TYPE == "vanilla_net":
                noise = torch.randn([batch_size, self.model_config_dict.G.LATENT_DIM], device=self.device)
                conditional = target
            elif self.model_config_dict.G.TYPE == "conv_net":
                noise = torch.randn([batch_size, self.model_config_dict.G.LATENT_DIM], device=self.device)
                conditional = F_torch.one_hot(target, num_classes).to(self.device).float()
            else:
                raise NotImplementedError(f"Model type `{self.model_config_dict.G.TYPE}` is not implemented.")

            ##############################################
            # (1) Update D network: max E(x)[log(D(x))] + E(z)[log(1- D(z))]
            # Start training the discriminator model
            ##############################################
            # Set discriminator gradients to zero.
            self.d_model.zero_grad()

            # Train with real.
            with torch.amp.autocast("cuda", enabled=self.device.type != "cpu"):
                real_output = self.d_model(inputs, conditional)
                d_loss_real = self.adv_criterion(real_output, real_label)
            # Call the gradient scaling function in the mixed precision API to
            # bp the gradient information of the fake samples
            self.scaler.scale(d_loss_real).backward()
            d_x = real_output.mean()

            # Train with fake.
            with torch.amp.autocast("cuda", enabled=self.device.type != "cpu"):
                fake = self.g_model(noise, conditional)
                fake_output = self.d_model(fake.detach(), conditional)
                d_loss_fake = self.adv_criterion(fake_output, fake_label)
            # Call the gradient scaling function in the mixed precision API to
            # bp the gradient information of the fake samples
            self.scaler.scale(d_loss_fake).backward()

            # Calculate the total discriminator loss value
            d_loss = d_loss_real + d_loss_fake
            d_g_z1 = fake_output.mean()
            self.scaler.step(self.d_optimizer)
            self.scaler.update()
            # Finish training the discriminator model

            ##############################################
            # (2) Update G network: min E(z)[log(1- D(z))]
            # Start training the generator model
            ##############################################
            # Initialize generator model gradients
            self.g_optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.device.type != "cpu"):
                fake_output = self.d_model(fake, conditional)
                g_loss = self.adv_criterion(fake_output, real_label)
            # Call the gradient scaling function in the mixed precision API to
            # bp the gradient information of the fake samples
            self.scaler.scale(g_loss).backward()
            # Encourage the generator to generate higher quality fake samples, making it easier to fool the discriminator
            self.scaler.step(self.g_optimizer)
            self.scaler.update()
            d_g_z2 = fake_output.mean()

            # Statistical accuracy and loss value for terminal data output
            batch_size = inputs.size(0)
            self.d_losses.update(d_loss.item(), batch_size)
            self.g_losses.update(g_loss.item(), batch_size)
            self.d_x_losses.update(d_x.item(), batch_size)
            self.d_g_z1_losses.update(d_g_z1.item(), batch_size)
            self.d_g_z2_losses.update(d_g_z2.item(), batch_size)

            # Calculate the time it takes to fully train a batch of data
            self.batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0 or i == self.num_train_batch - 1:
                iters = i + self.current_epoch * self.train_batch_size + 1
                self.tblogger.add_scalar("Train/D_Loss", d_loss.item(), iters)
                self.tblogger.add_scalar("Train/G_Loss", g_loss.item(), iters)
                self.tblogger.add_scalar("Train/D_x", d_x.item(), iters)
                self.tblogger.add_scalar("Train/D_G_z1", d_g_z1.item(), iters)
                self.tblogger.add_scalar("Train/D_G_z2", d_g_z2.item(), iters)
                progress.display(i + 1)

    def after_epoch(self):
        # update g lr
        self.g_lr_scheduler.step()
        self.d_lr_scheduler.step()

        self.eval_model()

        # save g model
        ckpt = {
            "model": deepcopy(self.g_model).half(),
            "ema": deepcopy(self.g_model).half(),
            "updates": None,
            "optimizer": self.g_optimizer.state_dict(),
            "scheduler": self.g_lr_scheduler.state_dict(),
            "epoch": self.current_epoch,
        }
        save_checkpoint(
            ckpt,
            self.save_checkpoint_dir,
            True,
            current_model_name=f"g_epoch_{self.current_epoch:04d}.pkl",
            best_model_name="g_best.pkl",
            last_model_name="g_last.pkl",
        )

        # save d model
        ckpt = {
            "model": deepcopy(self.d_model).half(),
            "ema": None,
            "updates": None,
            "optimizer": self.d_optimizer.state_dict(),
            "scheduler": self.d_lr_scheduler.state_dict(),
            "epoch": self.current_epoch,
        }
        save_checkpoint(
            ckpt,
            self.save_checkpoint_dir,
            True,
            current_model_name=f"d_epoch_{self.current_epoch:04d}.pkl",
            best_model_name="d_best.pkl",
            last_model_name="d_last.pkl",
        )

        del ckpt

    def eval_model(self) -> None:
        with torch.no_grad():
            # Switch model to eval mode.
            self.g_model.eval()
            fake_image = self.g_model(self.fixed_noise, self.fixed_conditional)
            fake_image_path = self.save_visual_dir.joinpath(f"epoch_{self.current_epoch:04d}.jpg")
            vutils.save_image(fake_image.detach(), fake_image_path, pad_value=255, normalize=True)
            LOGGER.info(f"Save fake image to `{fake_image_path}`")


def init_train_env(config_dict: DictConfig) -> [DictConfig, torch.device]:
    """Initialize the training environment.

    Args:
        config_dict (DictConfig): The configuration dictionary.

    Returns:
        [DictConfig, torch.device]: The configuration dictionary and the device for training
    """

    def _resume(config_dict: DictConfig, checkpoint_path: str):
        assert Path(checkpoint_path).is_file(), f"the checkpoint path is not exist: {checkpoint_path}"
        LOGGER.info(f"Resume training from the checkpoint file: `{checkpoint_path}`")
        resume_config_file_path = Path(checkpoint_path).parent.parent.joinpath(save_config_name)
        if resume_config_file_path.exists():
            config_dict = OmegaConf.load(resume_config_file_path)
        else:
            LOGGER.warning(f"Can not find the path of `{resume_config_file_path}`, will save exp log to {Path(checkpoint_path).parent.parent}")
            LOGGER.warning(f"In this case, make sure to provide configuration, such as datasets, batch size.")
            config_dict.TRAIN.SAVE_DIR = str(Path(checkpoint_path).parent.parent)
        return config_dict

    # Define the name of the configuration file
    save_config_name = "config.yaml"

    resume_g = config_dict.TRAIN.get("RESUME_G", "")
    resume_d = config_dict.TRAIN.get("RESUME_D", "")

    # Handle the resume training case
    if resume_g:
        config_dict = _resume(config_dict, resume_g)
        config_dict.TRAIN.RESUME_G = resume_g
    elif resume_d:
        config_dict = _resume(config_dict, resume_d)
        config_dict.TRAIN.RESUME_D = resume_d
    else:
        save_dir = Path(config_dict.TRAIN.OUTPUT_DIR).joinpath(Path(config_dict.EXP_NAME))
        config_dict.TRAIN.SAVE_DIR = str(increment_name(save_dir))
        Path(config_dict.TRAIN.SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # Select the device for training
    device = select_device(config_dict.DEVICE)

    # Set the random seed
    set_seed_everything(1 + config_dict.TRAIN.RANK, deterministic=(config_dict.TRAIN.RANK == -1))

    # Save the configuration
    save_config_path = Path(config_dict.TRAIN.SAVE_DIR).joinpath(save_config_name)
    OmegaConf.save(config_dict, save_config_path)

    return config_dict, device

from comet_ml import OfflineExperiment, Experiment
import torchvision
import numpy as np
import math
import torch
from torch import optim
from tqdm import tqdm
import torch.nn as nn
from torch import nn
import hydra
import os
import logging
import random
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloaders import MVU_Estimator_Brain, MVU_Estimator_Knees, MVU_Estimator_Stanford_Knees, MVU_Estimator_Abdomen
import multiprocessing
import PIL.Image
from torch.utils.data.distributed import DistributedSampler
from utils import *

from ncsnv2.models import get_sigmas
from ncsnv2.models.ema import EMAHelper
from ncsnv2.models.ncsnv2 import NCSNv2Deepest
import argparse

def normalize(gen_img, estimated_mvue):
    '''
        Estimate mvue from coils and normalize with 99% percentile.
    '''
    scaling = torch.quantile(estimated_mvue.abs(), 0.99)
    return gen_img * scaling

def unnormalize(gen_img, estimated_mvue):
    '''
        Estimate mvue from coils and normalize with 99% percentile.
    '''
    scaling = torch.quantile(estimated_mvue.abs(), 0.99)
    return gen_img / scaling


class LangevinOptimizer(torch.nn.Module):
    def __init__(self, config, logger, project_dir='./', experiment=None):
        super().__init__()

        self.config = config

        self.langevin_config = self._dict2namespace(self.config['langevin_config'])
        self.device = config['device']
        self.langevin_config.device = config['device']

        self.project_dir = project_dir
        self.score = NCSNv2Deepest(self.langevin_config).to(self.device)
        self.sigmas_torch = get_sigmas(self.langevin_config)

        self.sigmas = self.sigmas_torch.cpu().numpy()

        states = torch.load(os.path.join(project_dir, config['gen_ckpt']))#, map_location=self.device)

        self.score = torch.nn.DataParallel(self.score)

        self.score.load_state_dict(states[0], strict=True)
        if self.langevin_config.model.ema:
            ema_helper = EMAHelper(mu=self.langevin_config.model.ema_rate)
            ema_helper.register(self.score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(self.score)
        del states

        self.index = 0
        self.experiment = experiment
        self.logger = logger

    def _dict2namespace(self,langevin_config):
        namespace = argparse.Namespace()
        for key, value in langevin_config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    def _initialize(self):
        self.gen_outs = []

    # Centered, orthogonal ifft in torch >= 1.7
    def _ifft(self, x):
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        x = torch_fft.ifft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.fftshift(x, dim=(-2, -1))
        return x

    # Centered, orthogonal fft in torch >= 1.7
    def _fft(self, x):
        x = torch_fft.fftshift(x, dim=(-2, -1))
        x = torch_fft.fft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        return x

    def _sample(self, y):
        ref, mvue, maps, batch_mri_mask = y
        estimated_mvue = torch.tensor(
            get_mvue(ref.cpu().numpy(),
            maps.cpu().numpy()), device=ref.device)
        self.logger.info(f"Running {self.langevin_config.model.num_classes} steps of Langevin.")

        pbar = tqdm(range(self.langevin_config.model.num_classes), disable=(self.config['device'] != 0))
        pbar_labels = ['class', 'step_size', 'error', 'mean', 'max']
        step_lr = self.langevin_config.sampling.step_lr
        forward_operator = lambda x: MulticoilForwardMRI(self.config['orientation'])(torch.complex(x[:, 0], x[:, 1]), maps, batch_mri_mask)


        samples = torch.rand(y[0].shape[0], self.langevin_config.data.channels,
                                 self.config['image_size'][0],
                                 self.config['image_size'][1], device=self.device)

        with torch.no_grad():
            for c in pbar:
                if c <= self.config['start_iter']:
                    continue
                if c <= 1800:
                    n_steps_each = 3
                else:
                    n_steps_each = self.langevin_config.sampling.n_steps_each
                sigma = self.sigmas[c]
                labels = torch.ones(samples.shape[0], device=samples.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / self.sigmas[-1]) ** 2

                for s in range(n_steps_each):
                    noise = torch.randn_like(samples) * np.sqrt(step_size * 2)
                    # get score from model
                    p_grad = self.score(samples, labels)

                    # get measurements for current estimate
                    meas = forward_operator(normalize(samples, estimated_mvue))
                    # compute gradient, i.e., gradient = A_adjoint * ( y - Ax_hat )
                    # here A_adjoint also involves the sensitivity maps, hence the pointwise multiplication
                    # also convert to real value since the ``complex'' image is a real-valued two channel image
                    meas_grad = torch.view_as_real(torch.sum(self._ifft(meas-ref) * torch.conj(maps), axis=1) ).permute(0,3,1,2)
                    # re-normalize, since measuremenets are from a normalized estimate
                    meas_grad = unnormalize(meas_grad, estimated_mvue)
                    # convert to float incase it somehow became double
                    meas_grad = meas_grad.type(torch.cuda.FloatTensor)
                    meas_grad /= torch.norm( meas_grad )
                    meas_grad *= torch.norm( p_grad )
                    meas_grad *= self.config['mse']

                    # combine measurement gradient, prior gradient and noise
                    samples = samples + step_size * (p_grad - meas_grad) + noise

                    # compute metrics
                    metrics = [c, step_size, (meas-ref).norm(), (p_grad-meas_grad).abs().mean(), (p_grad-meas_grad).abs().max()]
                    update_pbar_desc(pbar, metrics, pbar_labels)
                    # if nan, break
                    if np.isnan((meas - ref).norm().cpu().numpy()):
                        return normalize(samples, estimated_mvue)
                if self.config['save_images']:
                    if (c+1) % self.config['save_iter'] ==0 :
                        img_gen = normalize(samples, estimated_mvue)
                        to_display = torch.view_as_complex(img_gen.permute(0, 2, 3, 1).reshape(-1, self.config['image_size'][0], self.config['image_size'][1], 2).contiguous()).abs()
                        if self.config['anatomy'] == 'brain':
                            # flip vertically
                            to_display = to_display.flip(-2)
                        elif self.config['anatomy'] == 'knees':
                            # flip vertically and horizontally
                            to_display = to_display.flip(-2)
                            to_display = to_display.flip(-1)
                        elif self.config['anatomy'] == 'stanford_knees':
                            # do nothing
                            pass
                        elif self.config['anatomy'] == 'abdomen':
                            # flip horizontally
                            to_display = to_display.flip(-1)
                        else:
                            pass
                        for i, exp_name in enumerate(self.config['exp_names']):
                            if self.config['repeat'] == 1:
                                file_name = f'{exp_name}_R={self.config["R"]}_{c}.jpg'
                                save_images(to_display[i:i+1], file_name, normalize=True)
                                if self.experiment is not None:
                                    self.experiment.log_image(file_name)
                            else:
                                for j in range(self.config['repeat']):
                                    file_name = f'{exp_name}_R={self.config["R"]}_sample={j}_{c}.jpg'
                                    save_images(to_display[j:j+1], file_name, normalize=True)
                                    if self.experiment is not None:
                                        self.experiment.log_image(file_name)

                        # uncomment below if you want to save intermediate samples, they are logged to CometML in the interest of saving space
                        # intermediate_out = samples
                        # intermediate_out.requires_grad = False
                        # self.gen_outs.append(intermediate_out)
                # if c>=0:
                #     break

        return normalize(samples, estimated_mvue)



    def sample(self, y):
        self._initialize()
        mvue = self._sample(y)

        outputs = []
        for i in range(y[0].shape[0]):
            outputs_ = {
                'mvue': mvue[i:i+1],
                # uncomment below if you want to return intermediate output
                # 'gen_outs': self.gen_outs
            }
            outputs.append(outputs_)
        return outputs

def mp_run(rank, config, project_dir, working_dir, files):
    if config['multiprocessing']:
        mp_setup(rank, config['world_size'])
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.INFO)
    logger = MpLogger(logger, rank)

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    logger.info(f'Logging to {working_dir}')
    if rank == 0 and not config['debug']:
        # uncomment the following to log the experiment offline
        # will need to add api key to see experiments online
        #api_key = None
        #project_name = config['anatomy']
        #experiment = Experiment(api_key,
        #                        project_name=project_name,
        #                        auto_output_logging='simple')
        project_name = config['anatomy']
        experiment = OfflineExperiment(
                                project_name=project_name,
                                auto_output_logging='simple',
                                offline_directory="./outputs")

        experiment.log_parameters(config)
        pretty(config)
    else:
        experiment = None

    config['device'] = rank
    # load appropriate dataloader
    if config['anatomy'] == 'knees':
        dataset = MVU_Estimator_Knees(files,
                            input_dir=config['input_dir'],
                            maps_dir=config['maps_dir'],
                            project_dir=project_dir,
                            image_size = config['image_size'],
                            R=config['R'],
                            pattern=config['pattern'],
                            orientation=config['orientation'])
    elif config['anatomy'] == 'stanford_knees':
        dataset = MVU_Estimator_Stanford_Knees(files,
                            input_dir=config['input_dir'],
                            maps_dir=config['maps_dir'],
                            project_dir=project_dir,
                            image_size = config['image_size'],
                            R=config['R'],
                            pattern=config['pattern'],
                            orientation=config['orientation'])
    elif config['anatomy'] == 'abdomen':
        dataset = MVU_Estimator_Abdomen(
                            input_dir=config['input_dir'],
                            maps_dir=config['maps_dir'],
                            project_dir=project_dir,
                            image_size = config['image_size'],
                            R=config['R'],
                            pattern=config['pattern'],
                            orientation=config['orientation'],
                            rotate=config['rotate'])

    elif config['anatomy'] == 'brain':
        dataset = MVU_Estimator_Brain(files,
                                input_dir=config['input_dir'],
                                maps_dir=config['maps_dir'],
                                project_dir=project_dir,
                                image_size = config['image_size'],
                                R=config['R'],
                                pattern=config['pattern'],
                                orientation=config['orientation'])
    else:
        raise NotImplementedError('anatomy not implemented, please write dataloader to process kspace appropriately')

    sampler = DistributedSampler(dataset, rank=rank, shuffle=True) if config['multiprocessing'] else None
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])


    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=config['batch_size'],
                                         sampler=sampler,
                                         shuffle=True if sampler is None else False)

    langevin_optimizer = LangevinOptimizer(config, logger, project_dir, experiment=experiment)
    if config['multiprocessing']:
        langevin_optimizer = DDP(langevin_optimizer, device_ids=[rank]).module
    langevin_optimizer.to(rank)

    for index, sample in enumerate(tqdm(loader)):
        '''
                    ref: one complex image per coil
                    mvue: one complex image reconstructed using the coil images and the sensitivity maps
                    maps: sensitivity maps for each one of the coils
                    mask: binary valued kspace mask
        '''

        ref, mvue, maps, mask = sample['ground_truth'], sample['mvue'], sample['maps'], sample['mask']
        # uncomment for meniscus tears
        # exp_name = sample['mvue_file'][0].split('/')[-1] + '|langevin|' + f'slide_idx_{sample["slice_idx"][0].item()}'
        # # if exp_name != 'file1000425.h5|langevin|slide_idx_22':
        # if exp_name != 'file1002455.h5|langevin|slide_idx_26':
        #     continue

        # move everything to cuda
        ref = ref.to(rank).type(torch.complex128)
        mvue = mvue.to(rank)
        maps = maps.to(rank)
        mask = mask.to(rank)
        estimated_mvue = torch.tensor(
            get_mvue(ref.cpu().numpy(),
            maps.cpu().numpy()), device=ref.device)


        exp_names = []
        for batch_idx in range(config['batch_size']):

            exp_name = sample['mvue_file'][batch_idx].split('/')[-1] + '|langevin|' + f'slide_idx_{sample["slice_idx"][batch_idx].item()}'
            exp_names.append(exp_name)
            print(exp_name)
            if config['save_images']:
                file_name = f'{exp_name}_R={config["R"]}_estimated_mvue.jpg'
                save_images(estimated_mvue[batch_idx:batch_idx+1].abs().flip(-2), file_name, normalize=True)
                if experiment is not None:
                    experiment.log_image(file_name)

                file_name = f'{exp_name}_input.jpg'
                save_images(mvue[batch_idx:batch_idx+1].abs().flip(-2), file_name, normalize=True)
                if experiment is not None:
                    experiment.log_image(file_name)

        langevin_optimizer.config['exp_names'] = exp_names
        if config['repeat'] > 1:
            repeat = config['repeat']
            ref, mvue, maps, mask, estimated_mvue = ref.repeat(repeat,1,1,1), mvue.repeat(repeat,1,1,1), maps.repeat(repeat,1,1,1), mask.repeat(repeat,1), estimated_mvue.repeat(repeat,1,1,1)
        outputs = langevin_optimizer.sample((ref, mvue, maps, mask))


        for i, exp_name in enumerate(exp_names):
            if config['repeat'] == 1:
                torch.save(outputs[i], f'{exp_name}_R={config["R"]}_outputs.pt')
            else:
                for j in range(config['repeat']):
                    torch.save(outputs[j], f'{exp_name}_R={config["R"]}_sample={j}_outputs.pt')

        # todo: delete after testing
        if index >= 0:
            break

    if config['multiprocessing']:
        mp_cleanup()

@hydra.main(config_path='configs')
def main(config):
    """ setup """

    working_dir = os.getcwd()
    project_dir = hydra.utils.get_original_cwd()

    folder_path = os.path.join(project_dir, config['input_dir'])
    if config['anatomy'] == 'stanford_knees':
        files = get_all_files(folder_path, pattern=f'*R{config["R"]}*.h5')
    else:
        files = get_all_files(folder_path, pattern='*.h5')

    if not config['multiprocessing']:
        mp_run(0, config, project_dir, working_dir, files)
    else:
        mp.spawn(mp_run,
                args=(config, project_dir, working_dir, files),
                nprocs=config['world_size'],
                join=True)


if __name__ == '__main__':
    main()

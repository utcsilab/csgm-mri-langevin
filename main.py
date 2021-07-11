from comet_ml import Experiment
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
from dataloaders import BrainMRIDataset, UndersampledRSS, MVU_Estimator, MVU_Estimator_Knees, MVU_Estimator_Stanford_Knees, MVU_Estimator_Abdomen
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
        # if config['image_size'][0] != config['image_size'][1]:
        #     raise Exception('Non-square images are not supported yet.')

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
        n_steps_each = self.langevin_config.sampling.n_steps_each
        step_lr = self.langevin_config.sampling.step_lr
        if self.config['no_maps']:
            forward_operator = lambda x: MulticoilForwardMRINoMaps()(torch.complex(x[:, 0], x[:, 1]), batch_mri_mask)
        else:
            forward_operator = lambda x: MulticoilForwardMRI(self.config['orientation'])(torch.complex(x[:, 0], x[:, 1]), maps, batch_mri_mask)


        if self.config['no_maps']:
            samples = torch.rand(self.num_coils,self.langevin_config.data.channels,
                                     self.config['image_size'][0],
                                     self.config['image_size'][1], device=self.device)
        else:
            samples = torch.rand(y[0].shape[0], self.langevin_config.data.channels,
                                     self.config['image_size'][0],
                                     self.config['image_size'][1], device=self.device)

        with torch.no_grad():
            for c in pbar:
                sigma = self.sigmas[c]
                labels = torch.ones(samples.shape[0], device=samples.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / self.sigmas[-1]) ** 2

                for s in range(n_steps_each):
                    noise = torch.randn_like(samples) * np.sqrt(step_size * 2)
                    grad = self.score(samples, labels)
                    meas = forward_operator(normalize(samples, estimated_mvue))
                    if self.config['no_maps']:
                        meas_grad = torch.view_as_real(self._ifft(meas-ref) /(sigma**2)).permute(0,3,1,2)
                    else:
                        meas_grad = self.config['mse'] * torch.view_as_real(torch.sum(self._ifft(meas-ref) * torch.conj(maps), axis=1) /(sigma**2)).permute(0,3,1,2)
                    meas_grad = unnormalize(meas_grad, estimated_mvue)
                    meas_grad = meas_grad.type(torch.cuda.FloatTensor)

                    samples = samples + step_size * (grad - meas_grad) + noise
                    # print("class: {}, step_size: {}, error {}, mean {}, max {}".format(c, step_size, (meas-ref).norm(), grad.abs().mean(), grad.abs().max()))
                    # print(normalize(samples,estimated_mvue).abs().max(), normalize(samples, estimated_mvue).abs().min(), mvue.abs().max(), mvue.abs().min())
                    metrics = [c, step_size, (meas-ref).norm(), (grad-meas_grad).abs().mean(), (grad-meas_grad).abs().max()]
                    update_pbar_desc(pbar, metrics, pbar_labels)
                    if np.isnan((meas - ref).norm().cpu().numpy()):
                        return normalize(samples, estimated_mvue)
                if self.config['save_images']:
                    if (c+1) % self.config['save_iter'] ==0 :
                        img_gen = normalize(samples, estimated_mvue)
                        to_display = torch.view_as_complex(img_gen.permute(0, 2, 3, 1).reshape(-1, self.config['image_size'][0], self.config['image_size'][1], 2).contiguous()).abs()
                        if self.config['no_maps']:
                            to_display = to_display.view(to_display.shape[0], 1, to_display.shape[1], to_display.shape[2])
                        if not self.config['is_knees']:
                            to_display = to_display.flip(-2)
                        else:
                            to_display = to_display.flip(-2)
                            to_display = to_display.flip(-1)
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
        api_key = 'Z86Oz16wGA1wEDpxzMEDIPDzJ'
        if config['is_knees']:
            project_name = 'langevin-mri-knees'
        elif config['is_abdomen']:
            project_name = 'langevin-mri-abdomen'
        elif config['is_stanford_knees']:
            project_name = 'langevin-mri-stanford-knees'
        else:
            project_name = 'langevin-mri-brains'
        experiment = Experiment(api_key,
                                project_name=project_name,
                                auto_output_logging='simple')
        experiment.log_parameters(config)
        pretty(config)
    else:
        experiment = None

    config['device'] = rank
    if config['is_knees']:
        dataset = MVU_Estimator_Knees(files,
                            raw_dir=config['raw_dir'],
                            maps_dir=config['maps_dir'],
                            project_dir=project_dir,
                            image_size = config['image_size'],
                            R=config['R'],
                            pattern=config['pattern'],
                            orientation=config['orientation'])
    elif config['is_stanford_knees']:
        dataset = MVU_Estimator_Stanford_Knees(files,
                            raw_dir=config['raw_dir'],
                            maps_dir=config['maps_dir'],
                            project_dir=project_dir,
                            image_size = config['image_size'],
                            R=config['R'],
                            pattern=config['pattern'],
                            orientation=config['orientation'])
    elif config['is_abdomen']:
        dataset = MVU_Estimator_Abdomen(
                            raw_dir=config['raw_dir'],
                            maps_dir=config['maps_dir'],
                            project_dir=project_dir,
                            image_size = config['image_size'],
                            R=config['R'],
                            pattern=config['pattern'],
                            orientation=config['orientation'],
                            rotate=config['rotate'])

    else:
        dataset = MVU_Estimator(files,
                                raw_dir=config['raw_dir'],
                                mvue_dir=config['mvue_dir'],
                                maps_dir=config['maps_dir'],
                                project_dir=project_dir,
                                image_size = config['image_size'],
                                R=config['R'],
                                pattern=config['pattern'],
                                orientation=config['orientation'])


    sampler = DistributedSampler(dataset, rank=rank, shuffle=False) if config['multiprocessing'] else None
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])


    # loader = torch.utils.data.DataLoader(dataset=dataset,
    #                                      batch_size=config['batch_size'],
    #                                      sampler=sampler,
    #                                      shuffle=True if sampler is None else False)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=config['batch_size'],
                                         sampler=sampler,
                                         shuffle=False)

    langevin_optimizer = LangevinOptimizer(config, logger, project_dir, experiment=experiment)
    if config['multiprocessing']:
        langevin_optimizer = DDP(langevin_optimizer, device_ids=[rank]).module
    langevin_optimizer.to(rank)

    for index, sample in enumerate(tqdm(loader)):
        '''
                    ref: one complex image per coil
                    mvue: one complex image reconstructed using the coil images and the sensitivity maps
                    maps: sensitivity maps for each one of the coils
        '''

        ref, mvue, maps, mask = sample['ground_truth'], sample['mvue'], sample['maps'], sample['mask']
        # uncomment for meniscus tears
        # exp_name = sample['mvue_file'][0].split('/')[-1] + '|langevin|' + f'slide_idx_{sample["slice_idx"][0].item()}'
        # # if exp_name != 'file1000425.h5|langevin|slide_idx_22':
        # if exp_name != 'file1002455.h5|langevin|slide_idx_26':
        #     continue
        xx = (ref==0)
        langevin_optimizer.num_coils = ref.shape[1]
        print(ref.shape[1])

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
        #if run:
        if config['no_maps']:
            outputs = langevin_optimizer.sample((ref[0], mvue, maps, mask[0]))
        else:
            outputs = langevin_optimizer.sample((ref, mvue, maps, mask))


        for i, exp_name in enumerate(exp_names):
            if config['repeat'] == 1:
                torch.save(outputs[i], f'{exp_name}_R={config["R"]}_outputs.pt')
            else:
                for j in range(config['repeat']):
                    torch.save(outputs[j], f'{exp_name}_R={config["R"]}_sample={j}_outputs.pt')

        if index >= 0:
            break

    if config['multiprocessing']:
        mp_cleanup()

@hydra.main(config_name='configs/run_langevin')
def main(config):
    """ setup """
    working_dir = os.getcwd()
    project_dir = hydra.utils.get_original_cwd()

    folder_path = os.path.join(project_dir, config['input_folder'])
    if config['is_stanford_knees']:
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

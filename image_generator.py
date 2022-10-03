# Copyright (c) 2022 Savoir-faire Linux Inc.
# This code is licensed under MIT license
import time
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import torch
from torch import autocast
from contextlib import contextmanager, nullcontext
from einops import rearrange
from PIL import Image
import numpy as np
import io

def load_model_from_config(opt, config):
    print(f"Loading Stable Diffusion model from {opt.ckpt}")
    pl_sd = torch.load(opt.ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return model.to(device)

class ImageGenerator:
    def init(self, opt):
        self.opt = opt
        self.config = OmegaConf.load(f"{opt.config}")
        self.model = load_model_from_config(opt, self.config.model)
        self.precision_scope = autocast if opt.precision=="autocast" else nullcontext
        if opt.plms:
            self.sampler = PLMSSampler(self.model)
        else:
            self.sampler = DDIMSampler(self.model)
        if opt.fixed_code:
            self.start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=self.model.device)
        else:
            self.start_code = None

    def generate_images(self, input: dict):
        prompt = input['text']
        print(f'Starting {self.opt.ddim_steps} steps Stable Diffusion for: "{prompt}"')
        data = [self.opt.n_samples * [prompt]]
        result = []

        with torch.no_grad():
            with self.precision_scope("cuda"):
                with self.model.ema_scope():
                    for n in range(self.opt.n_iter):
                        for prompts in data:
                            uc = None
                            if self.opt.scale != 1.0:
                                uc = self.model.get_learned_conditioning(self.opt.n_samples * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)
                            shape = [self.opt.C, self.opt.H // self.opt.f, self.opt.W // self.opt.f]
                            samples_ddim, _ = self.sampler.sample(S=self.opt.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=self.opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=self.opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=self.opt.ddim_eta,
                                                            x_T=self.start_code)

                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_checked_image_torch = x_samples_ddim
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                b = io.BytesIO()
                                Image.fromarray(x_sample.astype(np.uint8)).save(b, "jpeg")
                                result.append(b.getvalue())

        return input, result

#  based off of https://nn.labml.ai/diffusion/stable_diffusion/model/unet.html
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
# pip install ftfy regex tqdm
# pip install git+https://github.com/openai/CLIP.git
import clip
# pip install transformers accelerate sentencepiece
from transformers import T5Tokenizer, T5EncoderModel

from models.attention import SpatialTransformer
from models.unet import SinusoidalPositionEmbeddings
from typing import List, Optional


def normalization(channels):
    return GroupNorm32(32, channels)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class ResBlock(nn.Module):
    def __init__(self, channels: int, d_time_emb: int, *, out_channels: int = None):
        super().__init__()
        if out_channels is None:
            out_channels = channels

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_time_emb, out_channels)
        )

        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(0.),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        h = self.in_layers(x)
        time_emb = self.emb_layers(time_emb).type(h.dtype)
        h = h + time_emb[:, :, None, None]
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class TimestepEmbedSequential(nn.Sequential):
    '''Wrapper for sequential such that we can compose different modules that accept
    different inputs into one sequential object'''
    def forward(self, x, t_emb, cond=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class ConditionalUNet(nn.Module):
    def __init__(self,
                 *,
                 in_channels: int,
                 out_channels: int,
                 channels: int,
                 n_res_blocks: int,
                 attention_levels: List[int],
                 channel_multipliers: List[int],
                 n_heads: int,
                 tf_layers: int = 1,
                 d_cond: int = 768,
                 measure_dim = 2,
                 logvar: torch.tensor = None):
        super().__init__()

        # log-variance params to optimize saved here if we are doing latent diffusion
        self.logvar = torch.nn.Parameter(logvar, requires_grad=True)

        # embedding layer for the items to condition on
        self.cond_embed = nn.Sequential(
            nn.Linear(measure_dim, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond)
        )

        # determine dimensions
        self.channels = channels

        levels = len(channel_multipliers)

        d_time_embed = channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(channels),
            nn.Linear(channels, d_time_embed),
            nn.SiLU(),
            nn.Linear(d_time_embed, d_time_embed)
        )

        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(TimestepEmbedSequential(
            nn.Conv2d(in_channels, channels, 3, padding=1)
        ))

        input_block_channels = [channels]
        channels_list = [channels * m for m in channel_multipliers]

        # define the input blocks (first half of the UNet)
        for i in range(levels):
            for _ in range(n_res_blocks):
                layers = [ResBlock(channels, d_time_embed, out_channels=channels_list[i])]
                channels = channels_list[i]

                if i in attention_levels:
                    layers.append(SpatialTransformer(channels, n_heads, tf_layers, d_cond))

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(channels)

            if i != levels - 1:
                # add a DownSample to the end of every level
                self.input_blocks.append(TimestepEmbedSequential(DownSample(channels)))
                input_block_channels.append(channels)

        # define the middle part of the UNet
        self.middle_block = TimestepEmbedSequential(
            ResBlock(channels, d_time_embed),
            SpatialTransformer(channels, n_heads, tf_layers, d_cond)
        )

        # define the last half of the UNet
        self.output_blocks = nn.ModuleList()
        for i in reversed(range(levels)):
            for j in range(n_res_blocks + 1):
                layers = [ResBlock(channels + input_block_channels.pop(), d_time_embed, out_channels=channels_list[i])]
                channels = channels_list[i]

                if i in attention_levels:
                    layers.append(SpatialTransformer(channels, n_heads, tf_layers, d_cond))

                if i != 0 and j == n_res_blocks:
                    # add an UpSample to the end of every level
                    layers.append(UpSample(channels))

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, cond: Optional[torch.Tensor] = None):
        x_input_block = []
        t_emb = self.time_embed(time_steps)
        if cond is not None:
            cond = self.cond_embed(cond)[:, None, :]

        for module in self.input_blocks:
            x = module(x, t_emb, cond)
            x_input_block.append(x)

        x = self.middle_block(x, t_emb, cond)

        for module in self.output_blocks:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x, t_emb, cond)

        return self.out(x)

def get_dtype_from_transformer(self):
    return self.transformer.resblocks[0].mlp[0].weight.dtype

class LangConditionalUNet(ConditionalUNet):

    # Clip naming convention is model_type-size/patch_size
    def __init__(self, *args, language_model: str = 'clip', **kwargs):
        super().__init__(*args, **kwargs)
        self.language_model = language_model.lower()
        self.d_cond: int = self.cond_embed[-1].weight.shape[1]
        if self.language_model == 'clip':
            model, preprocess_img = clip.load("ViT-B/16", jit=False)
            del preprocess_img
            self.clip_model = model.float()
            # We can't just remove the visual transformer since the clip model uses it to know its own dtype
            del self.clip_model.visual
            setattr(type(self.clip_model), 'dtype', property(get_dtype_from_transformer))
            # We could patch the clip text projection instead, but why not stack more layers?
            # The clip encoder just uses the last latent state of the last token, so we can include a bias term here.
            self.clip_cond_projection = nn.Linear(self.clip_model.text_projection.shape[1],
                                                  self.d_cond, bias=True)
        elif self.language_model == 'flan-t5-small':
            self.t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
            self.t5_model = T5EncoderModel.from_pretrained("google/flan-t5-small", device_map="auto")
            # We're going to sum across the token dim (which varies based on label length), so don't use a bias.
            self.t5_hidden_projection = nn.Linear(self.t5_model.config.d_model, self.d_cond, bias=False)

    def text_to_cond(self, text: List[str]) -> torch.Tensor:
        if self.language_model == 'clip':
            tokenized = clip.tokenize(text).to(self.clip_cond_projection.weight.device)
            encoded = self.clip_model.encode_text(tokenized)
            return self.clip_cond_projection(encoded)
        elif self.language_model == 'flan-t5-small':
            tokenized = self.t5_tokenizer(text, return_tensors="pt", padding=True).input_ids.to(self.t5_hidden_projection.weight.device)
            hidden_states = self.t5_model(input_ids=tokenized).last_hidden_state
            projected = self.t5_hidden_projection(hidden_states)
            # Sum across token dimension (the projection will learn to sum the hidden states)
            return projected.sum(dim=-2)

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, cond: Optional[torch.Tensor] = None, cond_text: Optional[List[str]] = None):
        x_input_block = []
        t_emb = self.time_embed(time_steps)
        if cond is not None or cond_text is not None:
            if cond is None:
                cond = self.text_to_cond(cond_text)
            cond = cond[:, None, :]

        for module in self.input_blocks:
            x = module(x, t_emb, cond)
            x_input_block.append(x)

        x = self.middle_block(x, t_emb, cond)

        for module in self.output_blocks:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x, t_emb, cond)

        return self.out(x)

import sys 
sys.path.append(f'./comfyui-animatediff')
sys.path.append(f'./ComfyUI')


from einops import rearrange
import torch
import torch.nn as nn
import os 
module_dir = os.getcwd()

os.chdir(os.path.abspath('./ComfyUI'))
sys.argv=['']
import comfy
os.chdir(module_dir)


os.chdir(os.path.abspath('./comfyui-animatediff'))
from animatediff.motion_module import MotionModule, BlockType
os.chdir(module_dir)

os.chdir(f'./generative-models')
import sgm
os.chdir(module_dir)

os.chdir(f'./stablediffusion')
import ldm
os.chdir(module_dir)

import sys
import os
from torch import Tensor


from sgm.modules.diffusionmodules.openaimodel import TimestepBlock, SpatialTransformer
def new_TimestepEmbedSequential_forward_sdxl(
        self,
        x,
        emb,
        context=None,
        skip_time_mix=False,
        time_context=None,
        num_video_frames=None,
        time_context_cat=None,
        use_crossframe_attention_in_spatial_layers=False,
    ):
        
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif type(layer).__name__ == "VanillaTemporalModule":
                x = layer(x, encoder_hidden_states=context)
            else:
                x = layer(x)
        return x

from ldm.modules.diffusionmodules.openaimodel import TimestepBlock, SpatialTransformer
def new_TimestepEmbedSequential_forward(self, x, emb, context=None):
            # print('type(layer).__name__', type(layer).__name__)
            
            for layer in self:
                if isinstance(layer, TimestepBlock):
                    x = layer(x, emb)
                elif isinstance(layer, SpatialTransformer):
                    x = layer(x, context)
                elif type(layer).__name__ == "VanillaTemporalModule":
                  x = layer(x,encoder_hidden_states=context)
                  # x = layer(x, temb=None, encoder_hidden_states=context)
                else:
                    x = layer(x)
            return x

def new_TemporalTransformerBlock_forward(
          self,
          hidden_states,
          encoder_hidden_states=None,
          attention_mask=None,
          video_length=None,
      ):
          for attention_block, norm in zip(self.attention_blocks, self.norms):
              norm_hidden_states = norm(hidden_states)
              hidden_states = (
                  attention_block(
                      norm_hidden_states,
                      encoder_hidden_states=encoder_hidden_states if attention_block.is_cross_attention else None,
                      video_length=video_length,
                  )
                  + hidden_states
              )

          hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

          output = hidden_states
          return output

def new_TemporalTransformer3DModel_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
          batch, channel, height, weight = hidden_states.shape
          residual = hidden_states

          hidden_states = self.norm(hidden_states)
          inner_dim = hidden_states.shape[1]
          hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
          hidden_states = self.proj_in(hidden_states)

          # Transformer Blocks
          for block in self.transformer_blocks:
              hidden_states = block(
                  hidden_states,
                  encoder_hidden_states=encoder_hidden_states,
                  video_length=16,
              )

          # output
          hidden_states = self.proj_out(hidden_states)
          hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

          output = hidden_states + residual

          return output

  # Merge from https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved
def get_encoding_max_len(mm_state_dict: dict[str, Tensor]) -> int:
      # use pos_encoder.pe entries to determine max length - [1, {max_length}, {320|640|1280}]
      for key in mm_state_dict.keys():
          if key.endswith("pos_encoder.pe"):
              return mm_state_dict[key].size(1)  # get middle dim
      raise ValueError(f"No pos_encoder.pe found in mm_state_dict")

def has_mid_block(mm_state_dict: dict[str, Tensor]):
      # check if keys contain mid_block
      for key in mm_state_dict.keys():
          if key.startswith("mid_block."):
              return True
      return False

  #taken from https://github.com/ArtVentureX/comfyui-animatediff
class MotionWrapper(nn.Module):
    def __init__(self, mm_type: str, encoding_max_len: int = 24, is_v2=False, is_sdxl=False, is_v3=False):
        super().__init__()
        self.mm_type = mm_type
        self.is_v2 = is_v2
        self.is_sdxl = is_sdxl
        self.is_v3 = is_v3
        self.hack_gn = not (is_sdxl or is_v3)
        self.is_hotshot = False

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        self.mid_block = None
        self.encoding_max_len = encoding_max_len

        channels = [320, 640, 1280, 1280] if not is_sdxl else [320, 640, 1280]
        for c in channels:
            self.down_blocks.append(MotionModule(c, BlockType.DOWN, encoding_max_len=encoding_max_len))
            self.up_blocks.insert(0,MotionModule(c, BlockType.UP, encoding_max_len=encoding_max_len))
        if is_v2:
            self.mid_block = MotionModule(1280, BlockType.MID, encoding_max_len=encoding_max_len)

    @classmethod
    def from_state_dict(cls, mm_state_dict: dict[str, Tensor], mm_type: str, is_sdxl=False, is_v3=False):
        encoding_max_len = get_encoding_max_len(mm_state_dict)
        is_v2 = has_mid_block(mm_state_dict)

        mm = cls(mm_type, encoding_max_len=encoding_max_len, is_v2=is_v2, is_sdxl=is_sdxl, is_v3=is_v3)
        mm.load_state_dict(mm_state_dict, strict=False)
        return mm

    def set_video_length(self, video_length: int):
        for block in self.down_blocks:
            block.set_video_length(video_length)
        for block in self.up_blocks:
            block.set_video_length(video_length)
        if self.mid_block is not None:
            self.mid_block.set_video_length(video_length)

def inject_motion_module_to_unet(diffusion_model, mm):
    if diffusion_model.mm_injected:
      print('mm already injected. exiting')
      return

    #insert motion modules depending on surrounding layers
    for i in range(12 if not mm.is_sdxl else 9):
        a, b = divmod(i, 3)
        if type(diffusion_model.input_blocks[i][-1]).__name__ not in ["Downsample","Conv2d"]:
            # print('down', i,a,b)
            diffusion_model.input_blocks[i].append(mm.down_blocks[a].motion_modules[b-1])

        if type(diffusion_model.output_blocks[i][-1]).__name__ == "Upsample":
            # print('up', i,a,b)
            diffusion_model.output_blocks[i].insert(-1, mm.up_blocks[a].motion_modules[b])
        else:
            # print('up', i,a,b)
            diffusion_model.output_blocks[i].append(mm.up_blocks[a].motion_modules[b])
    if mm.is_v2:
      # pass
      diffusion_model.middle_block.insert(-1, mm.mid_block.motion_modules[0])
    elif mm.hack_gn:

            if mm.is_hotshot:
                from sgm.modules.diffusionmodules.util import GroupNorm32
            else:
                from ldm.modules.diffusionmodules.util import GroupNorm32
            diffusion_model.gn32_original_forward = GroupNorm32.forward
            gn32_original_forward = diffusion_model.gn32_original_forward

            def groupnorm32_mm_forward(self, x):
                x = rearrange(x, "(b f) c h w -> b c f h w", b=2)
                x = gn32_original_forward(self, x)
                x = rearrange(x, "b c f h w -> (b f) c h w", b=2)
                return x

            GroupNorm32.forward = groupnorm32_mm_forward

    diffusion_model.mm_injected = True

def eject_motion_module_from_unet(diffusion_model, mm):
    if not diffusion_model.mm_injected:
      print('mm not injected. exiting')
      return
    #remove motion modules depending on surrounding layers
    for i in range(12 if not mm.is_sdxl else 9):
        a, b = divmod(i, 3)
        if type(diffusion_model.input_blocks[i][-1]).__name__ == 'VanillaTemporalModule':
            diffusion_model.input_blocks[i].pop(-1)

        if type(diffusion_model.output_blocks[i][-2]).__name__ == 'VanillaTemporalModule':
            diffusion_model.output_blocks[i].pop(-2)
        elif type(diffusion_model.output_blocks[i][-1]).__name__ == 'VanillaTemporalModule':
            diffusion_model.output_blocks[i].pop(-1)
    if mm.is_v2:
      if type(diffusion_model.middle_block[-2]).__name__ == 'VanillaTemporalModule':
        # pass
        diffusion_model.middle_block.pop(-2)

    elif mm.hack_gn:
            if mm.is_hotshot:
                from sgm.modules.diffusionmodules.util import GroupNorm32
            else:
                from ldm.modules.diffusionmodules.util import GroupNorm32
            GroupNorm32.forward = diffusion_model.gn32_original_forward
            diffusion_model.gn32_original_forward = None

    diffusion_model.mm_injected = False

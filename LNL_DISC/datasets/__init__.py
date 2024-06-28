from .noise_datasets import cifar_dataloader
from .clothing1M import clothing_dataloader
from .webvision import webvision_dataloader
from .food101N import food101N_dataloader
from .animal10N import animal10N_dataloader
from .tiny_imagenet import tiny_imagenet_dataloader
from .BBBC import BBBC_dataloader
from .CP import CP_dataloader
from .dg_clothing import dg_clothing_dataloader
from .dg_vlcs import dg_vlcs_dataloader
from .dg_cp import dg_cp_dataloader

__all__ = ('cifar_dataloader', 'clothing_dataloader', 'dg_clothing_dataloader', 'dg_vlcs_dataloader', 'webvision_dataloader', 
           'food101N_dataloader', 'tiny_imagenet_dataloader', 'animal10N_dataloader', 'BBBC_dataloader', 'CP_dataloader', 'dg_cp_dataloader')

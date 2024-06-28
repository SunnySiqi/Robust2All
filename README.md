# Robust2All: A Benchmark for Robustness to In-Domain Noise and Out-of-Domain Generalization
Implementation of the paper "Robust2All: A Benchmark for Robustness to In-Domain Noise and Out-of-Domain Generalization"

## Dependencies
The code has been written using Python3 (3.10.4), run `pip install -r requirements.txt` to install relevant python packages.

## Dataset Download
VLCS & DomainNet: python VLCS_DomainNet_download.py --data_dir=/my/datasets/path\
Robust2All-Fashion [Clothing1M]: Please follow the instructions at: https://github.com/Cysu/noisy_label\
Robust2All-Fashion [FashionMNIST]: Will be downloaded automatically to the given root dirctory in the dataloaders.\
CHAMMI-CP: https://zenodo.org/records/7988357\

## Training
### DG Methods
+ ERM++: code at 'DG_ERMplusplus'. running script at test.sh. Data_path is needed in 'DG_ERMplusplus/data/dataset.py'.
+ MIRO: code at 'DG_MIRO_SWAD_SAGM_LNL_ELR'. running script at test.sh. Set the argument --algorthm 'MIRO'. set 'SWAD=False' at 'DG_MIRO_SWAD_SAGM_LNL_ELR/config.yaml'. Data_path is needed in 'DG_MIRO_SWAD_SAGM_LNL_ELR/domainbed/datasets/datasets.py'.
+ SWAD: code at 'DG_MIRO_SWAD_SAGM_LNL_ELR'. running script at test.sh. Set the argument --algorthm 'ERM'. set 'SWAD=True' at 'DG_MIRO_SWAD_SAGM_LNL_ELR/config.yaml'. Data_path is needed in 'DG_MIRO_SWAD_SAGM_LNL_ELR/domainbed/datasets/datasets.py'.
+ MIRO+SWAD: code at 'DG_MIRO_SWAD_SAGM_LNL_ELR'. running script at test.sh. Set the argument --algorthm 'MIRO'. set 'SWAD=True' at 'DG_MIRO_SWAD_SAGM_LNL_ELR/config.yaml'. Data_path is needed in 'DG_MIRO_SWAD_SAGM_LNL_ELR/domainbed/datasets/datasets.py'.
+ SAGM: code at 'DG_MIRO_SWAD_SAGM_LNL_ELR'. running script at test.sh. Set the argument --algorthm 'SAGM_DG'. set 'SWAD=False' at 'DG_MIRO_SWAD_SAGM_LNL_ELR/config.yaml'. Data_path is needed in 'DG_MIRO_SWAD_SAGM_LNL_ELR/domainbed/datasets/datasets.py'.
+ SAGM+SWAD: code at 'DG_MIRO_SWAD_SAGM_LNL_ELR'. running script at test.sh. Set the argument --algorthm 'SAGM_DG'. set 'SWAD=True' at 'DG_MIRO_SWAD_SAGM_LNL_ELR/config.yaml'. Data_path is needed in 'DG_MIRO_SWAD_SAGM_LNL_ELR/domainbed/datasets/datasets.py'.

### LNL Methods
+ ELR: code at 'DG_MIRO_SWAD_SAGM_LNL_ELR', running script at test.sh. To run with ELR only, set the argument --algorthm 'ERM', set 'SWAD=False' at 'DG_MIRO_SWAD_SAGM_LNL_ELR/config.yaml' and "hparams["elr"] = True" at "DG_MIRO_SWAD_SAGM_LNL_ELR/domainbed/hparams_registry.py". elr_beta and elr_lambda can also be assigned at "DG_MIRO_SWAD_SAGM_LNL_ELR/domainbed/hparams_registry.py". Data_path is needed in 'DG_MIRO_SWAD_SAGM_LNL_ELR/domainbed/datasets/datasets.py'.
+ DISC: code at 'LNL_DISC', running scripts in 'shs'. Data_path is needed in the scripts.
+ UNICON: code at 'LNL_UNICON', running scripts in 'scripts'. Data_path is needed in the scripts.

### DG+LNL Methods
+ ERM++ + ELR: code at 'DG_ERMplusplus'. running script at test.sh (need to add argument --elr, --elr_beta, --elr_lambda). Data_path is needed in 'DG_ERMplusplus/data/dataset.py'.
+ MIRO+UNICON: code at 'LNL_UNICON', running scripts in 'scripts'. Data_path is needed in the scripts. set 'SWAD=False' at 'LNL_UNICON/domainbed/config.yaml'.
+ MIRO+SWAD+UNICON: code at 'LNL_UNICON', running scripts in 'scripts'. Data_path is needed in the scripts. set 'SWAD=True' at 'LNL_UNICON/domainbed/config.yaml'.
+ MIRO+ELR: code at 'DG_MIRO_SWAD_SAGM_LNL_ELR', running script at test.sh. Set the argument --algorthm 'MIRO', set 'SWAD=False' at 'DG_MIRO_SWAD_SAGM_LNL_ELR/config.yaml' and "hparams["elr"] = True" at "DG_MIRO_SWAD_SAGM_LNL_ELR/domainbed/hparams_registry.py". elr_beta and elr_lambda can also be assigned at "DG_MIRO_SWAD_SAGM_LNL_ELR/domainbed/hparams_registry.py". Data_path is needed in 'DG_MIRO_SWAD_SAGM_LNL_ELR/domainbed/datasets/datasets.py'.
+ SWAD+ELR: code at 'DG_MIRO_SWAD_SAGM_LNL_ELR', running script at test.sh. Set the argument --algorthm 'ERM', set 'SWAD=True' at 'DG_MIRO_SWAD_SAGM_LNL_ELR/config.yaml' and "hparams["elr"] = True" at "DG_MIRO_SWAD_SAGM_LNL_ELR/domainbed/hparams_registry.py". elr_beta and elr_lambda can also be assigned at "DG_MIRO_SWAD_SAGM_LNL_ELR/domainbed/hparams_registry.py". Data_path is needed in 'DG_MIRO_SWAD_SAGM_LNL_ELR/domainbed/datasets/datasets.py'.
+ MIRO+SWAD+ELR: code at 'DG_MIRO_SWAD_SAGM_LNL_ELR', running script at test.sh. Set the argument --algorthm 'MIRO', set 'SWAD=True' at 'DG_MIRO_SWAD_SAGM_LNL_ELR/config.yaml' and "hparams["elr"] = True" at "DG_MIRO_SWAD_SAGM_LNL_ELR/domainbed/hparams_registry.py". elr_beta and elr_lambda can also be assigned at "DG_MIRO_SWAD_SAGM_LNL_ELR/domainbed/hparams_registry.py". Data_path is needed in 'DG_MIRO_SWAD_SAGM_LNL_ELR/domainbed/datasets/datasets.py'.

## Reference Code
 - https://github.com/JackYFL/DISC
 - https://github.com/nazmul-karim170/UNICON-Noisy-Label
 - https://github.com/shengliu66/ELR
 - https://github.com/piotr-teterwak/erm_plusplus
 - https://github.com/kakaobrain/miro
 - https://github.com/Wang-pengfei/SAGM



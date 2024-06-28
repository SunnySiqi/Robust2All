#python train_all.py vlcs_miro --algorithm MIRO --dataset VLCS --model resnet50 --steps 5000  > vlcs_miro.out
#python train_all.py vlcs_swad --algorithm ERM --dataset VLCS --model resnet50 --steps 5000  > vlcs_swad.out
#python train_all.py vlcs_miro_elr --algorithm MIRO --dataset VLCS --model resnet50 --steps 5000  > vlcs_miro.out
#python train_all.py vlcs_swad_elr --algorithm ERM --dataset VLCS --model resnet50 --steps 5000  > vlcs_swad.out
#python train_all.py vlcs_miro_swad_elr --algorithm MIRO --dataset VLCS --model resnet50 --steps 5000 > vlcs_elr_miro_swad.out
#python train_all.py cp_elr_miro_swad --algorithm MIRO --dataset CP --model convnext --steps 5000 --test_envs 1 2 3 > CP_elr_miro_0.3_ssh.out
#python train_all.py cp_swad --algorithm ERM --dataset CP --model convnext --steps 5000 --test_envs 1 2 3 > CP_swad_ssh.out
#python train_all.py cp_elr_miro --algorithm MIRO --dataset CP --model convnext --steps 5000 --test_envs 1 2 3 > CP_elr_miro.out
#python train_all.py domainnet_asym0.2 --algorithm ERM --dataset domainnet --model resnet50 --steps 15000 --add_noise --noise_ratio 0.2 --noise_type asym > domainnet_asym0.2.out 
python train_all.py clothing_miro_elr --algorithm MIRO --dataset Clothing --test_envs 1 --model resnet50 --steps 15000 > clothing_miro_elr.out
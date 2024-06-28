module load miniconda
conda deactivate
conda activate /projectnb/ivc-ml/siqiwang/anaconda3/envs/env

model_name='DISC_vlcs_caltech'
gpuid='0'
seed=123
save_path='./logs/'
#Todo: fill in the data_path
#data_path=
config_path='./configs/DISC_dgvlcs_caltech.py'
dataset='dg_vlcs'
num_classes=5

python main.py -c=$config_path --save_path=$save_path\
               --gpu=$gpuid --model_name=$model_name \
               --root=$data_path --seed=$seed\
               --dataset=$dataset --num_classes=$num_classes > disc_dg_vlcs_caltech.out
            

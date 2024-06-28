module load miniconda
conda deactivate
conda activate /projectnb/ivc-ml/siqiwang/anaconda3/envs/env

model_name='DISC_dgcp_task2'
gpuid='0'
seed=123
save_path='./logs/'
#Todo: fill in the data_path
#data_path=
config_path='./configs/DISC_dgcp_task2.py'
dataset='dg_CP'
num_classes=4

python main.py -c=$config_path --save_path=$save_path\
               --gpu=$gpuid --model_name=$model_name \
               --root=$data_path --seed=$seed\
               --dataset=$dataset --num_classes=$num_classes > disc_dg_cp_task2.out
            

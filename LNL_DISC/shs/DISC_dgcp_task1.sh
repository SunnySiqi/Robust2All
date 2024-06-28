model_name='DISC_dgcp_task1'
gpuid='0'
seed=123
save_path='./logs/'
#Todo: fill in the data_path
#data_path=
config_path='./configs/DISC_dgcp_task1.py'
dataset='dg_CP'
num_classes=4

python main.py -c=$config_path --save_path=$save_path\
               --gpu=$gpuid --model_name=$model_name \
               --root=$data_path --seed=$seed\
               --dataset=$dataset --num_classes=$num_classes > disc_dg_cp_task1.out
            

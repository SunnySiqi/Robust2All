model_name='DISC'
gpuid='0'
seed=123
save_path='./logs/'
#Todo: fill in the data_path
#data_path=
config_path='./configs/DISC_dgclothing.py'
dataset='dg_clothing'
num_classes=5

python main.py -c=$config_path --save_path=$save_path\
               --gpu=$gpuid --model_name=$model_name \
               --root=$data_path --seed=$seed\
               --dataset=$dataset --num_classes=$num_classes > disc_dg_clothing.out
            

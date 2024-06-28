algorithm = 'DISC'
# dataset param
dataset = 'dg_CP'
input_channel = 5
num_classes = 4
root = '../../baseline_fine/FINE_official-master/dynamic_selection/dir_to_data/CP/'
noise_type = 'sym'
percent = 0.8
seed = 123
loss_type = 'ce'
# model param
model1_type = 'ConvNeXt'
model2_type = 'none'
# train param
gpu = '0'
batch_size = 32
lr = 0.0002
epochs = 30
num_workers = 4
epoch_decay_start = 10
alpha = 5.0
# result param
save_result = True
# test and train domains
test_domain = "Task_two"
train_domain = ["Train", "Task_one", "Task_three"]
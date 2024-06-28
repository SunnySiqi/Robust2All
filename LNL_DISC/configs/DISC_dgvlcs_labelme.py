algorithm = 'DISC'
# dataset param
dataset = 'dg_vlcs'
input_channel = 3
num_classes = 5
root = '/projectnb/ivc-ml/piotrt/data/VLCS'
noise_type = 'sym'
percent = 0.8
seed = 123
loss_type = 'ce'
# model param
model1_type = 'resnet50'
model2_type = 'none'
# train param
gpu = '0'
batch_size = 32
lr = 0.005
epochs = 100
num_workers = 8
epoch_decay_start = 30
alpha = 5.0
# result param
save_result = True
# test and train domains
test_domain = ["LabelMe"]
train_domain = ["Caltech101", "SUN09", "VOC2007"]
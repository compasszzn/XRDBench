task=spg
model=IUCrj_CNN
epochs=100
lr=0.01
batch_size=64
gpu=3

python -u main.py --model $model --task $task --epochs $epochs --lr $lr --batch_size $batch_size --gpu $gpu
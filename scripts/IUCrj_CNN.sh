task=crysystem
model=IUCrj_CNN
epochs=100
lr=0.1
batch_size=64
gpu=7

python -u main.py --model $model --task $task --epochs $epochs --lr $lr --batch_size $batch_size --gpu $gpu
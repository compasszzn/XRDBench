task=crysystem
model=FCN
epochs=100
lr=0.0005
batch_size=64
gpu=7

python -u main.py --model $model --task $task --epochs $epochs --lr $lr --batch_size $batch_size --gpu $gpu
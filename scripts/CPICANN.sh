task=spg
model=CPICANN
epochs=100
lr=0.00025
batch_size=128
gpu=2

python -u main.py --model $model --task $task --epochs $epochs --lr $lr --batch_size $batch_size --gpu $gpu
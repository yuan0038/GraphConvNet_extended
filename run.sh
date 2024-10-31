export CUDA_VISIBLE_DEVICES=0,1,2,3
node_num=4
batch_size=256  #per gpu
model='graphconvnet_ti'   # change to the model you want to train:graphconvnet_ti graphconvnet_s  graphconvnetp_ti graphconvnetp_s

### train
python -u -m torch.distributed.launch --master_port 22577 --nproc_per_node=${node_num} train.py your/imagenet/root/path  --model  ${model} --sched cosine --epochs 300 --opt adamw -j 8 --warmup-lr 1e-6 --mixup .8 --cutmix 1.0  --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --warmup-epochs 20 --opt-eps 1e-8 --repeated-aug  --remode pixel --reprob 0.25  --amp --lr 2e-3 --weight-decay .05 --drop 0 --drop-path .1 -b ${batch_size}   --output ./output/${model}mlp_${batch_size}x${node_num}   >./${model}_${batch_size}x${node_num}.log 2>&1 &

### eval
# python train.py your/imagenet/root/path --model=${model} -b=${batch_size} --pretrain_path /path/to/pretrained/model/ --evaluate
echo "CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12359  main_hash.py \
    --cfg configs/resnet152.yaml  --data-path /data/fine-grained/cub-200-2011/CUB_200_2011/ \
    --batch-size 64 --accumulation-steps 2  --hash_bit 48 --tag hash_48_resnet152_clsm1 \
    --valid_rank 0 --amp-opt-level O0"
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12359  main_hash.py \
    --cfg configs/resnet152.yaml  --data-path /data/fine-grained/cub-200-2011/CUB_200_2011/ \
    --batch-size 64 --accumulation-steps 2  --hash_bit 48 --tag hash_48_resnet152_clsm1 \
    --valid_rank 0 --amp-opt-level O0

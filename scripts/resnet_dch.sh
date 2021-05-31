echo "CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12358  main_hash.py \
    --cfg configs/resnet152.yaml  --data-path /data/fine-grained/cub-200-2011/CUB_200_2011/ \
    --batch-size 64 --accumulation-steps 2  --hash_bit 32 --tag hash_32_resnet152_clsm1 \
    --valid_rank 0 --amp-opt-level O0"
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12358  main_hash.py \
    --cfg configs/resnet50.yaml  --data-path /data/ludi/datasets/ \
    --batch-size 64 --accumulation-steps 2  --hash_bit 64 --tag dogs_hash_64_resnet50_clsm1 \
    --valid_rank 0 --amp-opt-level O0 --dataset Dogs

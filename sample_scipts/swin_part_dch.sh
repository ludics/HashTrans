echo "CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12360 main_hash.py \
    --cfg configs/swin_base_patch4_window7_224.yaml  --data-path /data/fine-grained/cub-200-2011/CUB_200_2011/ \
    --batch-size 64 --accumulation-steps 2 --hash_bit 48 --tag hash_48_swin_base_clsm1 \
    --resume downloads/swin_tiny_patch4_window7_224.pth --pretrained downloads/swin_tiny_patch4_window7_224.pth \
    --valid_rank 0 --amp-opt-level O0"
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node 2 --master_port 13405 main_hash_part.py \
    --cfg configs/swin_part_tiny_patch4_window7_224.yaml  --data-path /data1/ludi/datasets/Dogs \
    --batch-size 64 --accumulation-steps 2 --hash_bit 32 --tag dogs_hash_32_swin_part_tiny_clsm1 \
    --resume downloads/swin_tiny_patch4_window7_224.pth --pretrained downloads/swin_tiny_patch4_window7_224.pth \
    --valid_rank 1 --amp-opt-level O0

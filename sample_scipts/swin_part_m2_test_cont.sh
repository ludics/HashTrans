echo "CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12360 main_hash.py \
    --cfg configs/swin_base_patch4_window7_224.yaml  --data-path /data/fine-grained/cub-200-2011/CUB_200_2011/ \
    --batch-size 64 --accumulation-steps 2 --hash_bit 48 --tag hash_48_swin_base_clsm1 \
    --resume downloads/swin_tiny_patch4_window7_224.pth --pretrained downloads/swin_tiny_patch4_window7_224.pth \
    --valid_rank 0 --amp-opt-level O0"
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node 1 --master_port 13480 main_hash_part.py \
    --cfg configs/swin_part_tiny_patch4_window7_224.yaml  --data-path /data/ludi/datasets/Dogs \
    --batch-size 64 --accumulation-steps 2 --hash_bit $1 --tag just_eval_2 \
    --resume $2 \
    --valid_rank 0 --amp-opt-level O0 --att_size 4 --eval_cont --output output/test --dataset Dogs

# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 13481 main_hash_part.py \
#     --cfg configs/swin_part_m2_tiny_patch4_window7_224.yaml  --data-path /data/fine-grained/cub-200-2011/CUB_200_2011/ \
#     --batch-size 64 --accumulation-steps 2 --hash_bit 32 --tag just_eval_2 \
#     --resume output/swin_gwl_m2/swin_tiny_patch4_window7_224/hash_32_swin_part_tiny_clsm1/ckpt_best.pth \
#     --valid_rank 0 --amp-opt-level O0 --att_size 4 --eval --output output/test &

# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node 1 --master_port 13482 main_hash_part.py \
#     --cfg configs/swin_part_m2_tiny_patch4_window7_224.yaml  --data-path /data/fine-grained/cub-200-2011/CUB_200_2011/ \
#     --batch-size 64 --accumulation-steps 2 --hash_bit 48 --tag just_eval_3 \ 
#     --resume output/swin_gwl_m2/swin_tiny_patch4_window7_224/hash_48_swin_part_tiny_clsm1/ckpt_best.pth \
#     --valid_rank 0 --amp-opt-level O0 --att_size 4 --eval --output output/test &

# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node 1 --master_port 13483 main_hash_part.py \
#     --cfg configs/swin_part_m2_tiny_patch4_window7_224.yaml  --data-path /data/fine-grained/cub-200-2011/CUB_200_2011/ \
#     --batch-size 64 --accumulation-steps 2 --hash_bit 64 --tag just_eval_4 \
#     --resume output/swin_gwl_m2/swin_tiny_patch4_window7_224/hash_64_swin_part_tiny_clsm1/ckpt_best.pth \
#     --valid_rank 0 --amp-opt-level O0 --att_size 4 --eval --output output/test &

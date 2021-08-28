echo "CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node 1 --master_port 13405 main_hash_part.py \
    --cfg configs/swin_part_tiny_patch4_window7_224.yaml  --data-path /data1/ludi/datasets/StanfordCars \
    --batch-size 64 --accumulation-steps 2 --hash_bit 32 --tag cars_hash_32_swin_part_tiny_clsm1 \
    --resume downloads/swin_tiny_patch4_window7_224.pth --pretrained downloads/swin_tiny_patch4_window7_224.pth \
    --valid_rank 1 --amp-opt-level O0 --dataset Cars"
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node 1 --master_port 13405 main_hash_part.py \
    --cfg configs/swin_part_tiny_patch4_window7_224.yaml  --data-path /data1/ludi/datasets/StanfordCars \
    --batch-size 32 --accumulation-steps 2 --hash_bit 32 --tag cars_hash_32_swin_part_tiny_clsm1 \
    --resume downloads/swin_tiny_patch4_window7_224.pth --pretrained downloads/swin_tiny_patch4_window7_224.pth \
    --valid_rank 0 --amp-opt-level O0 --dataset Cars

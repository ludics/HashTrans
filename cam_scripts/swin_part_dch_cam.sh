echo "CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12358  main_hash.py \
    --cfg configs/resnet152.yaml  --data-path /data/fine-grained/cub-200-2011/CUB_200_2011/ \
    --batch-size 64 --accumulation-steps 2  --hash_bit 32 --tag hash_32_resnet152_clsm1 \
    --valid_rank 0 --amp-opt-level O0"
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12358  main_cam.py \
    --cfg configs/swin_part_m2_tiny_patch4_window7_224.yaml  --data-path /data/ludi/datasets/cub-200-2011/CUB_200_2011 \
    --batch-size 64 --accumulation-steps 2  --hash_bit 48 --tag cam_hash_48_resnet50_clsm1 \
    --valid_rank 0 --amp-opt-level O0 --att_size 4 --dataset CUB_200_2011 \
    --resume ../24G_output/output/swin_gwl_m2/swin_tiny_patch4_window7_224/hash_48_swin_part_m2_tiny_clsm1/ckpt_best.pth --make_cam

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12370  main_adsh.py  --cfg configs/resnet152_adsh.yaml    --data-path /data/fine-grained/cub-200-2011/CUB_200_2011/ --batch-size 64 --accumulation-steps 2  --tag hash_64_resnet152_adsh --hash_bit 64 --gamma 200.0 --lambd_cls 0.0  --valid_rank 0 --amp-opt-level O0
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345  main_hash.py  --cfg configs/resnet152.yaml  --data-path /data/fine-grained/cub-200-2011/CUB_200_2011/ --batch-size 64 --accumulation-steps 2  --tag hash_64_resnet152 --hash_bit 64 --gamma 20.0 --lambd 0.1 --lambd_cls 0.1 --valid_rank 0

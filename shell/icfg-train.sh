#export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7

python -m torch.distributed.run --nproc_per_node=4 --rdzv_endpoint=127.0.0.1:29502 \
Retrieval.py \
--config configs/PS_icfg_pedes.yaml \
--output_dir output/icfg-pedes/train \
--checkpoint /data/ALBEF/ALBEF.pth \
--eval_mAP

export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=4,5,6,7

python -m torch.distributed.run --nproc_per_node=4 --rdzv_endpoint=127.0.0.1:29501 \
Retrieval.py \
--config configs/PS_rstp_reid.yaml \
--output_dir output/rstp-reid/evaluation/ \
--checkpoint ../rasa_checkpoint/rasa_rstp_checkpoint.pth \
--eval_mAP \
--evaluate

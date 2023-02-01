CUDA_VISIBLE_DEVICES=6 python run_attn_diff_semi.py --dataset Freebase_m --run_num 10 --hidden-dim 8 --num-heads 16 --num-layers 5 --edge-feats 8 --cuda_device 0 --lam 1.0 --sample 3 --tem 0.5

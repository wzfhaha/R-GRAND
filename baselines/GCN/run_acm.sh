# for 5 classes
#CUDA_VISIBLE_DEVICES=0 python run_gcn.py --trials 50 --dataset ACM --cuda_device 0 --lr 5e-4 --l2norm 5e-4 -e 1000  --patience 50  --run_num 5 --run_num_seed 3 --hidden_dim 512 --num_layers 2 --train_num 20 --dropout 0.3 --path ../../data/ --feats-type 2
CUDA_VISIBLE_DEVICES=0 python run_gcn.py --trials 50 --dataset ACM --cuda_device 0 --lr 5e-4 --l2norm 5e-4 -e 1000  --patience 50  --run_num 5 --run_num_seed 3 --hidden_dim 32 --num_layers 2 --train_num 5 --dropout 0.7 --path ../../data/ --feats-type 2

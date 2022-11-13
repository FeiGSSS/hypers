python main_superpixel.py --data_name MNIST\
                          --model_name GIN\
                          --num_layers 4\
                          --hid_dim 110\
                          --dropout 0\
                          --residule True\
                          --batch_norm True\
                          --eps_train True\
                          --readout sum\
                          --seed 41\
                          --epochs 1000\
                          --batch_size 128\
                          --lr 0.001\
                          --lr_reduce_factor 0.5\
                          --lr_schedule_patience 10\
                          --min_lr 1e-5\
                          --wd 0.0\
                          --cuda_id 1
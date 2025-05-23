python main_mosi.py \
    --train \
    --use_adam_drop \
    --fusion_method concat \
    --q_base 0.5 \
    --lam 0.9 \
    --p_exe 0.9 \
    --data_path "/kaggle/input/dataset-mosi/mosi_data.pkl" \
    --batch_size 64 \
    --lr 2e-3 \
    --cls_lr 6e-7 \
    --rou 1.6 \
    --lamda 0.7

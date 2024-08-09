model_name=QURNN

python -u prediction_for_qurnn.py  \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_96_96 \
        --model QURNN \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96  \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --d_model 16 \
        --d_ff 32 \
        --des 'Exp' \
        --itr 1 \
        --top_k 5 \
        --activation relu  \
        --learning_rate 0.001 \
        --hidden_dim '[64]' \
        --cell_dim '[64]'

python -u prediction_for_qurnn.py  \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_96_192 \
        --model QURNN \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 192  \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --d_model 16 \
        --d_ff 32 \
        --des 'Exp' \
        --itr 1 \
        --top_k 5 \
        --activation relu  \
        --learning_rate 0.001 \
        --hidden_dim '[64]' \
        --cell_dim '[64]'

python -u prediction_for_qurnn.py  \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_96_336 \
        --model QURNN \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 336  \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --d_model 16 \
        --d_ff 32 \
        --des 'Exp' \
        --itr 1 \
        --top_k 5 \
        --activation relu  \
        --learning_rate 0.001 \
        --hidden_dim '[64]' \
        --cell_dim '[64]'


python -u prediction_for_qurnn.py  \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_96_720 \
        --model QURNN \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 720  \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --d_model 16 \
        --d_ff 32 \
        --des 'Exp' \
        --itr 1 \
        --top_k 5 \
        --activation relu  \
        --learning_rate 0.001 \
        --hidden_dim '[64]' \
        --cell_dim '[64]'



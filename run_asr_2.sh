. path.sh

# Quantized ASR Experiments
python main_asr.py --filename 'Q1_15-5L-512' --num_epochs 30 \
    --path_dataset $PATH_TIMIT  --seed 2 --step 0 --val 1 --run_through 1 --bestmodel 'per' \
    --cache_size 5000 --nfilt 40 --phn 48 --cla_type 'GRU' --cla_layers 5 --cla_size 512 --bidirectional 0 \
    --opt 'ADAM' --lr 3e-4 --decay_rate 0.8 --decay_epoch 5 --decoder 'beam' --beam_width 10 \
    --quantize 1 --m 1 --n 15 --cuda 1 || exit 1;

python main_asr.py --filename 'Q1_15-5L-512' --num_epochs 30 \
    --path_dataset $PATH_TIMIT  --seed 2 --step 3 --val 1 --run_through 1 --bestmodel 'vloss' \
    --cache_size 5000 --nfilt 40 --phn 48 --cla_type 'GRU' --cla_layers 5 --cla_size 512 --bidirectional 0 \
    --opt 'ADAM' --lr 3e-4 --decay_rate 0.8 --decay_epoch 5 --decoder 'beam' --beam_width 10 \
    --quantize 1 --m 1 --n 15 --cuda 1 || exit 1;

python main_asr.py --filename 'Q1_7-5L-512' --num_epochs 30 \
    --path_dataset $PATH_TIMIT  --seed 2 --step 2 --val 1 --run_through 1 --bestmodel 'per' \
    --cache_size 5000 --nfilt 40 --phn 48 --cla_type 'GRU' --cla_layers 5 --cla_size 512 --bidirectional 0 \
    --opt 'ADAM' --lr 3e-4 --decay_rate 0.8 --decay_epoch 5 --decoder 'beam' --beam_width 10 \
    --quantize 1 --m 1 --n 7 --cuda 1 || exit 1;

python main_asr.py --filename 'Q1_7-5L-512' --num_epochs 30 \
    --path_dataset $PATH_TIMIT  --seed 2 --step 3 --val 1 --run_through 1 --bestmodel 'vloss' \
    --cache_size 5000 --nfilt 40 --phn 48 --cla_type 'GRU' --cla_layers 5 --cla_size 512 --bidirectional 0 \
    --opt 'ADAM' --lr 3e-4 --decay_rate 0.8 --decay_epoch 5 --decoder 'beam' --beam_width 10 \
    --quantize 1 --m 1 --n 7 --cuda 1 || exit 1;

python main_asr.py --filename 'Q1_15-5L-512B' --num_epochs 30 \
    --path_dataset $PATH_TIMIT  --seed 2 --step 2 --val 1 --run_through 1 --bestmodel 'per' \
    --cache_size 5000 --nfilt 40 --phn 48 --cla_type 'GRU' --cla_layers 5 --cla_size 512 --bidirectional 1 \
    --opt 'ADAM' --lr 3e-4 --decay_rate 0.8 --decay_epoch 5 --decoder 'beam' --beam_width 10 \
    --quantize 1 --m 1 --n 15 --cuda 1 || exit 1;

python main_asr.py --filename 'Q1_15-5L-512B' --num_epochs 30 \
    --path_dataset $PATH_TIMIT  --seed 2 --step 3 --val 1 --run_through 1 --bestmodel 'vloss' \
    --cache_size 5000 --nfilt 40 --phn 48 --cla_type 'GRU' --cla_layers 5 --cla_size 512 --bidirectional 1 \
    --opt 'ADAM' --lr 3e-4 --decay_rate 0.8 --decay_epoch 5 --decoder 'beam' --beam_width 10 \
    --quantize 1 --m 1 --n 15 --cuda 1 || exit 1;

python main_asr.py --filename 'Q1_7-5L-512B' --num_epochs 30 \
    --path_dataset $PATH_TIMIT  --seed 2 --step 2 --val 1 --run_through 1 --bestmodel 'per' \
    --cache_size 5000 --nfilt 40 --phn 48 --cla_type 'GRU' --cla_layers 5 --cla_size 512 --bidirectional 1 \
    --opt 'ADAM' --lr 3e-4 --decay_rate 0.8 --decay_epoch 5 --decoder 'beam' --beam_width 10 \
    --quantize 1 --m 1 --n 7 --cuda 1 || exit 1;

python main_asr.py --filename 'Q1_7-5L-512B' --num_epochs 30 \
    --path_dataset $PATH_TIMIT  --seed 2 --step 3 --val 1 --run_through 1 --bestmodel 'vloss' \
    --cache_size 5000 --nfilt 40 --phn 48 --cla_type 'GRU' --cla_layers 5 --cla_size 512 --bidirectional 1 \
    --opt 'ADAM' --lr 3e-4 --decay_rate 0.8 --decay_epoch 5 --decoder 'beam' --beam_width 10 \
    --quantize 1 --m 1 --n 7 --cuda 1 || exit 1;

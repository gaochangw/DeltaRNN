. path.sh

#VAD Experiments
python main_vad.py --filename 'W1' --num_epochs 20 \
    --path_dataset $PATH_QUT_NOISE  --seed 2 --step 0 --run_through 1 \
    --nfilt 40 --window_size 1 --frame_size 0.1 --frame_stride 0.1\
    --opt 'ADAM' --lr 1e-4 --batch_size 512 \
    --quantize 0 --m 1 --n 15 --cuda 1 || exit 1;

python main_vad.py --filename 'W3' --num_epochs 20 \
    --path_dataset $PATH_QUT_NOISE  --seed 2 --step 0 --run_through 1 \
    --nfilt 40 --window_size 3 --frame_size 0.1 --frame_stride 0.1 \
    --opt 'ADAM' --lr 1e-4 --batch_size 512 \
    --quantize 0 --m 1 --n 15 --cuda 1 || exit 1;

python main_vad.py --filename 'W5' --num_epochs 20 \
    --path_dataset $PATH_QUT_NOISE  --seed 2 --step 0 --run_through 1 \
    --nfilt 40 --window_size 5 --frame_size 0.1 --frame_stride 0.1\
    --opt 'ADAM' --lr 1e-4 --batch_size 512 \
    --quantize 0 --m 1 --n 15 --cuda 1 || exit 1;

python main_vad.py --filename 'W1_swap' --num_epochs 20  --swap 1 \
    --path_dataset $PATH_QUT_NOISE  --seed 2 --step 0 --run_through 1 \
    --nfilt 40 --window_size 1 --frame_size 0.1 --frame_stride 0.1\
    --opt 'ADAM' --lr 1e-4 --batch_size 512 \
    --quantize 0 --m 1 --n 15 --cuda 1 || exit 1;

python main_vad.py --filename 'W3_swap' --num_epochs 20 --swap 1 \
    --path_dataset $PATH_QUT_NOISE  --seed 2 --step 0 --run_through 1 \
    --nfilt 40 --window_size 3 --frame_size 0.1 --frame_stride 0.1 \
    --opt 'ADAM' --lr 1e-4 --batch_size 512 \
    --quantize 0 --m 1 --n 15 --cuda 1 || exit 1;

python main_vad.py --filename 'W5_swap' --num_epochs 20 --swap 1 \
    --path_dataset $PATH_QUT_NOISE  --seed 2 --step 0 --run_through 1 \
    --nfilt 40 --window_size 5 --frame_size 0.1 --frame_stride 0.1\
    --opt 'ADAM' --lr 1e-4 --batch_size 512 \
    --quantize 0 --m 1 --n 15 --cuda 1 || exit 1;
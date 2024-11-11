CUDA_VISIBLE_DEVICES=0

python train.py --precision 32 --run_name baseline-fp32
python train.py --precision 16 --run_name baseline-fp16
python train.py --precision bf16 --run_name baseline-bf16
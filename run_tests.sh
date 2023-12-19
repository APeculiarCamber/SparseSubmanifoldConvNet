#python sp_training.py --grid_size 64 --model_type inplace --step 8 --epochs 12 --lr 0.001
#python sp_training.py --grid_size 32 --model_type inplace --step 8 --epochs 12 --lr 0.001

#python sp_training.py --grid_size 64 --model_type atomic --step 8 --epochs 12 --lr 0.001
python sp_training.py --grid_size 32 --model_type atomic --step 8 --epochs 12 --lr 0.001
python sp_training.py --grid_size 64 --model_type atomic --step 8 --epochs 12 --lr 0.001

python sp_training.py --grid_size 32 --model_type dense --step 8 --epochs 12 --lr 0.001
python sp_training.py --grid_size 64 --model_type dense --step 8 --epochs 12 --lr 0.001

python sp_training.py --grid_size 32 --model_type inplace --step 8 --epochs 12 --lr 0.001
python sp_training.py --grid_size 64 --model_type inplace --step 8 --epochs 12 --lr 0.001

# python sp_training.py --grid_size 128 --model_type atomic --step 8 --epochs 12 --lr 0.001

#python sp_training.py --grid_size 128 --model_type inplace --step 8 --epochs 12 --lr 0.001

#python sp_training.py --grid_size 128 --model_type atomic --step 8 --epochs 12 --lr 0.001
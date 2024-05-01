srun --nodes=4 --cpus-per-task=8 --gres=gpu:4 --mem=32G python -u main_moco.py './datasets/ROD2021/sequences/train' --epochs 20 --world-size 4 --workers 8 --dist-url 'env://' --multiprocessing-distributed --batch-size 128 --moco-dim 128


the following command works properly on nodes with `TITAN X/XP` in a multi-processing fashion (i.e. multiple gpus per node)
```bash
srun --nodes=2 --cpus-per-task=4 --gpus-per-node=2 --mem=32G python -u main_moco.py './datasets/ROD2021/sequences/train' --epochs 20 --world-size 4 --workers 4 --dist-url 'env://' --multiprocessing-distributed --batch-size 32 --moco-dim 128
```
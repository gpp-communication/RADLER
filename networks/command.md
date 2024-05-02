srun --nodes=4 --cpus-per-task=8 --gres=gpu:4 --mem=32G python -u main_moco.py './datasets/ROD2021/sequences/train' --epochs 20 --world-size 4 --workers 8 --dist-url 'env://' --multiprocessing-distributed --batch-size 128 --moco-dim 128


the following command works properly on nodes with `p6000 + rtx 5000` with queue having a size of 65535
```bash
srun --nodes=2 --cpus-per-task=4 --gres=gpu:2,VRAM=20G --mem=32G python -u main_moco.py './datasets/CRUW' --epochs 20 --world-size 2 --workers 4 --dist-url 'env://' --multiprocessing-distributed --batch-size 8 --checkpoints-dir './logs/checkpoints/ssl/test' --moco-dim 128 --moco-k 65535
```


the following command works properly on nodes with `a40 + rtx 8000`
```bash
srun --nodes=2 --cpus-per-task=4 --gres=gpu:2,VRAM=48G --mem=32G python -u main_moco.py './datasets/CRUW' --epochs 20 --world-size 2 --workers 4 --dist-url 'env://' --multiprocessing-distributed --batch-size 16 --checkpoints-dir './logs/checkpoints/ssl/test' --moco-dim 1024 --moco-k 16384
```
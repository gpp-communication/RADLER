### A GPU with 20GB RAM can handle 2 data at one time **without** resuming from any checkpoints
```bash
srun --nodes=2 --cpus-per-task=8 --gres=gpu:2,VRAM=20G --mem=32G python -u main_moco.py './datasets/CRTUM/data_cluster_1_2/pretext' --epochs 20 --world-size 2 --workers 8 --dist-url 'env://' --multiprocessing-distributed --batch-size 8 
```

### A GPU with 20GB RAM can handle 2 data at one time **with** resuming from any checkpoints
```bash
srun --nodes=2 --cpus-per-task=8 --gres=gpu:2,VRAM=26G --mem=32G python -u main_moco.py './datasets/CRTUM/data_cluster_1_2/pretext' --epochs 20 --world-size 2 --workers 8 --dist-url 'env://' --multiprocessing-distributed --batch-size 8 --resume logs/checkpoints/ssl/checkpoint_0001.pth.tar
```


### A GPU with 48GB RAM (`a40 + rtx 8000`) can handle 16 data at one time **without** resuming from any checkpoints.
```bash
srun --nodes=2 --cpus-per-task=8 --gres=gpu:2,VRAM=48G --mem=32G python -u main_moco.py './datasets/CRTUM/data_cluster_1_2/pretext' --epochs 20 --world-size 2 --workers 8 --dist-url 'env://' --multiprocessing-distributed --batch-size 64
```

### A GPU with 48GB RAM (`a40 + rtx 8000`) can handle 8 data at one time **with** resuming from any checkpoints.
```bash
srun --nodes=2 --cpus-per-task=8 --gres=gpu:2,VRAM=48G --mem=32G python -u main_moco.py './datasets/CRTUM/data_cluster_1_2/pretext' --epochs 20 --world-size 2 --workers 8 --dist-url 'env://' --multiprocessing-distributed --batch-size 32 --resume logs/checkpoints/ssl/checkpoint_0001.pth.tar
```
# RADLER: Radar Object Detection Leveraging Semantic 3D City Models and Self-Supervised Radar-Image Learning

## Branches
- **`main`**: Contains the model for experiments with the CRCTUM dataset.
- **`ssl-rodnet-encoders`**: Includes code for contrastive SSL training of RODNet-CDC's encoder using the CRUW dataset.

## Program Structure
- **`main_moco.py`**: Entry point for the pretext task of contrastive SSL.
- **`main_downstream_train.py`**: Entry point for the radar object detection training downstream task.
- **`main_downstream_test.py`**: Entry point for testing the radar object detection downstream task.

### Execution Order
1. **Pretext Task**: `main_moco`
2. **Downstream Training**: `main_downstream_train`
3. **Downstream Testing**: `main_downstream_test`

**Note:** In the `main` branch, the pretrained encoder weights are frozen during the downstream task. In the `ssl-rodnet-encoders` branch, these weights are not frozen.

## Datasets

### CRCTUM Dataset
Download the dataset [here](#).

### CRUW Dataset
Refer to [RODNet](https://github.com/yizhou-wang/RODNet) for downloading the dataset.

Additional steps for using the CRUW dataset with the `ssl-rodnet-encoders` branch:
- Retain only the first chirp of each radar frame, renaming files to align with the `%06d.npy` format used by the images.
- Manually split the training and testing datasets (since the CRUW dataset does not include a test set). Refer to [T-RODNet](https://github.com/Zhuanglong2/T-RODNet?tab=readme-ov-file#prepare-data-for-rodnet) for details on dataset splitting.

## Running the Program
Refer to the [MoCo](https://github.com/facebookresearch/moco) project for the environmental setup. A `Slurm` environment is assumed, with the following example scripts provided:

### Pretext Training on the CRCTUM Dataset
```bash
#!/bin/bash
#SBATCH --job-name="moco-pretext"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4,VRAM=48G
#SBATCH --mem=64G
#SBATCH --time=50:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.err
export NCCL_P2P_DISABLE=1
srun python main_moco.py './datasets/CRCTUM/data_cluster_1_2/pretext' --epochs 150 --world-size 1 --workers 12 --dist-url 'env://' --multiprocessing-distributed --batch-size 64 --learning-rate 0.000001 --moco-k 512 --save-frequency 10 --checkpoints-dir ./logs/checkpoints/CRCTUM/
```

### Downstream Training on the CRCTUM Dataset
```bash
#!/bin/bash
#SBATCH --job-name="moco-downstream"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,VRAM=48G
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.err
srun python main_downstream_train.py './datasets/CRTUM/data_cluster_1_2/downstream/sequences/train' --epochs 60 --world-size 1 --workers 8 --dist-url 'env://' --multiprocessing-distributed --batch-size 32 --learning-rate 0.001 --checkpoints-dir ./logs/checkpoints/downstream/fuse-sdm/ --pretrained ./logs/checkpoints/ssl/checkpoint_file_from_pretrain.pth.tar --save-frequency 10 --fuse-semantic-depth-tensor
#srun python main_downstream_train.py './datasets/CRTUM/data_cluster_1_2/downstream/sequences/train' --epochs 60 --world-size 1 --workers 8 --dist-url 'env://' --multiprocessing-distributed --batch-size 32 --learning-rate 0.001 --checkpoints-dir ./logs/checkpoints/downstream/no-sdm/ --pretrained ./logs/checkpoints/ssl/checkpoint_file_from_pretrain.pth.tar --save-frequency 10
```

### Downstream Testing on the CRCTUM Dataset
```bash
#!/bin/bash
#SBATCH --job-name="moco-downstream"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,VRAM=20G
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.err
srun python main_downstream_test.py './datasets/CRTUM/data_cluster_1_2/downstream/sequences/test' --world-size 1 --workers 8 --dist-url 'env://' --multiprocessing-distributed --batch-size 64 --results-dir ./logs/results/CRTUM/ --pretrained ./logs/checkpoints/CRTUM/downstream/training-32-0.001-fuse_semantic_depth_tensor_False/checkpoint_from_downstream_task_training.pth.tar
```

**Note:** The scripts for the `ssl-rodnet-encoders` branch follow a similar structure but are adapted for the CRUW dataset.

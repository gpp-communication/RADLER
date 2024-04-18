from CRTUM_dataset import CRTUMDataset
from torch.utils.data import DataLoader


def CRTUM_dataloader(root, batch_size, num_workers=4, image_transform=None,
                     radar_frames_transform=None, pin_memory=True):
    dataset = CRTUMDataset(root, image_transform, radar_frames_transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dataloader


if __name__ == '__main__':
    dataloader = CRTUM_dataloader('/Users/chengxuyuan/Downloads/ROD2021/sequences/train_test', batch_size=64)
    print(dataloader)

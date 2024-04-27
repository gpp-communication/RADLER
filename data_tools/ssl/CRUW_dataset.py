import numpy as np
from PIL import Image
from CRTUM_dataset import CRTUMDataset


class CRUWDataset(CRTUMDataset):
    def __getitem__(self, idx):
        img_path = self.df['images'][idx]
        radar_path = self.df['radar_frames'][idx]
        image = Image.open(img_path).convert('RGB')
        radar_frame = np.load(radar_path)
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.radar_transform is not None:
            radar_frame = self.radar_transform(radar_frame)
        return image, radar_frame


if __name__ == '__main__':
    CRUW_dataset = CRUWDataset('../../datasets/CRUW')
    print(CRUW_dataset.__len__())


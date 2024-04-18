from imports import *


class RoadDataset(Dataset):
    def __init__(self, df, mode, root=root_dir):
        self.df = df
        self.rescale_size = 224
        self.mode = mode
        
    def __len__(self):
        return len(self.df)
    
    def aug(self, image, mask):
        if self.mode == "train":
            transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=[0.5], std=[0.25]),
                ToTensorV2(),
            ])
        else:
            transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.5], std=[0.25]),
                ToTensorV2(),
            ])
        transformed = transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']
        
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['tiff_image_path']
        mask_path = self.df.iloc[idx]['tif_label_path']
        image = Image.open(img_path)
        label = Image.open(mask_path)

        bw_image_array = np.array(label)
        color_image_array = np.array(image)

        top_left_x = random.randint(0, 1500 - 224)
        top_left_y = random.randint(0, 1500 - 224)
            
        bw_square = bw_image_array[top_left_y:top_left_y + 224, top_left_x:top_left_x + 224]
            
        color_square = color_image_array[top_left_y:top_left_y + 224, top_left_x:top_left_x + 224, :]

        white_pixels = np.sum(bw_square == 1)
        total_pixels = np.prod(bw_square.shape)
        while (white_pixels / total_pixels) < 0.02:
            top_left_x = random.randint(0, 1500 - 224)
            top_left_y = random.randint(0, 1500 - 224)
            bw_square = bw_image_array[top_left_y:top_left_y + 224, top_left_x:top_left_x + 224]
            color_square = color_image_array[top_left_y:top_left_y + 224, top_left_x:top_left_x + 224, :]
            white_pixels = np.sum(bw_square == 255)
            total_pixels = np.prod(bw_square.shape)

        image, mask = self.aug(color_square, bw_square)
        
        return image, mask
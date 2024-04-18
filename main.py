from imports import *
from dataset import RoadDataset
from train import train
from test import test 

def main():
    root_dir = root_dir
    metadata_path = metadata_path

    data_df = pd.read_csv(metadata_path)
    train_df = data_df.loc[data_df['split'] == 'train']
    val_df = data_df.loc[data_df['split'] == 'val'] 
    test_df = data_df.loc[data_df['split'] == 'test']

    train_dataset = RoadDataset(train_df, "train")
    val_dataset = RoadDataset(val_df, "val")
    test_dataset = RoadDataset(test_df, "test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)

    train(train_loader, val_loader)
    test(test_loader) 

if __name__ == '__main__':
    main()
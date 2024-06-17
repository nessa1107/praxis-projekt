from torch.utils.data import DataLoader
from dataLoaderFloodNet import DataLoaderFloodNet


batch_size = 4
num_classes = 10

train_dataset = DataLoaderFloodNet(folder_path=r'FloodNet\train')
val_dataset = DataLoaderFloodNet(folder_path=r'FloodNet\val')
test_dataset = DataLoaderFloodNet(folder_path=r'FloodNet\test')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

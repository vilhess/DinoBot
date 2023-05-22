import torch
import torchvision
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from dataset import DinoDataset, transformer
import pandas as pd
from sklearn.model_selection import train_test_split

key_frame = pd.read_csv('labels_dino.csv')
train, test = train_test_split(key_frame, test_size=0.2)

full_set = DinoDataset(root_dir="captures",
                       dataframe=key_frame, transform=transformer)
full_loader = DataLoader(full_set, batch_size=1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = 'efficientnet_v2_s.pth'

saved_model = torchvision.models.efficientnet_v2_s()
saved_model.classifier = torch.nn.Linear(in_features=1280, out_features=2)
saved_model.load_state_dict(torch.load(path))
saved_model = saved_model.to(device)
saved_mode = saved_model.eval()

if __name__ == "__main__":

    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(full_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = saved_model(images)
            predicted = torch.softmax(outputs, dim=1).argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f'\\n Accuracy of the network on the test images: {100 * correct // total} %')

from torch.utils.data import Dataset
import torch
import os
import requests


class HW3Dataset(Dataset):
    url = 'https://technionmail-my.sharepoint.com/:u:/g/personal/ploznik_campus_technion_ac_il/EUHUDSoVnitIrEA6ALsAK1QBpphP5jX3OmGyZAgnbUFo0A?download=1'

    def __init__(self, root, transform=None, pre_transform=None):
        super(HW3Dataset, self).__init__()
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.data = self.load_data()

    def load_data(self):
        data_file = os.path.join(self.root, 'data.pt')
        if not os.path.exists(data_file):
            self.download()
        return torch.load(data_file)

    def download(self):
        file_url = self.url.replace(' ', '%20')
        response = requests.get(file_url)

        if response.status_code != 200:
            raise Exception(f"Failed to download the file, status code: {response.status_code}")

        # Check if directory exists and if not, create it
        os.makedirs(self.root, exist_ok=True)

        with open(os.path.join(self.root, 'data.pt'), 'wb') as f:
            f.write(response.content)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data

import torch
from torch.utils.data import Dataset, DataLoader
import ase.db

class ASEDataset(Dataset):
    def __init__(self, db_path, split):
        self.db = ase.db.connect(db_path)
        self.split = split
        print("load data from ",db_path)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        row = self.db.get(idx+1)  # ASE db indexing starts from 1
        
        # Extract relevant data from the row
        # atoms = row.toatoms()
        # element = atoms.get_chemical_symbols()
        latt_dis = eval(getattr(row, 'latt_dis'))
        intensity = eval(getattr(row, 'intensity'))
        spg = eval(getattr(row, 'tager'))[0]
        crysystem = eval(getattr(row, 'tager'))[1]

        # Convert to tensors
        # tensor_element = torch.tensor(element, dtype=torch.int64)  # Assuming you want to encode elements as integers
        tensor_latt_dis = torch.tensor(latt_dis, dtype=torch.float32)
        tensor_intensity = torch.tensor(intensity, dtype=torch.float32)
        tensor_spg = torch.tensor(spg, dtype=torch.int64)
        tensor_crysystem = torch.tensor(crysystem, dtype=torch.int64)

        return {
            # 'element': tensor_element,
            'latt_dis': tensor_latt_dis,
            'intensity': tensor_intensity,
            'spg': tensor_spg,
            'crysystem': tensor_crysystem,
            'split': self.split
        }
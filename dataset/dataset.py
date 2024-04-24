import torch
from torch.utils.data import Dataset, DataLoader
import ase.db
import json

class ASEDataset(Dataset):
    def __init__(self, db_paths,encode_element):
        with open('/home/zinanzheng/project/github/XRDBench/element/element/CGCNN_atom_emb.json' , 'r') as file:
            self.cgcnn_emb = json.load(file)
        self.db_paths = db_paths
        self.encode_element = encode_element
        print(db_paths)
        self.dbs = [ase.db.connect(db_path) for db_path in db_paths]
        print("Loaded data from:", db_paths)

    def __len__(self):
        total_length = sum(len(db) for db in self.dbs)
        return total_length

    def __getitem__(self, idx):
        
        cumulative_length = 0
        for i, db in enumerate(self.dbs):
            if idx < cumulative_length + len(db):
                # Adjust the index to the range of the current database
                adjusted_idx = idx - cumulative_length
                row = db.get(adjusted_idx + 1)  # ASE db indexing starts from 1
                if self.encode_element:
                    atoms = row.toatoms()
                    element = atoms.get_chemical_symbols()
                    element_encode = self.symbol_to_atomic_number(element)
                    element_value = []
                    for code in element_encode:
                        value = self.cgcnn_emb[str(code)]
                        element_value.append(value)
                    element_value=torch.mean(torch.tensor(element_value, dtype=torch.float32),dim=0)
                # Extract relevant data from the row
                # latt_dis = eval(getattr(row, 'latt_dis'))
                intensity = eval(getattr(row, 'intensity'))
                spg = eval(getattr(row, 'tager'))[0]
                crysystem = eval(getattr(row, 'tager'))[1]

                # Convert to tensors
                # tensor_latt_dis = torch.tensor(latt_dis, dtype=torch.float32)
                tensor_intensity = torch.tensor(intensity, dtype=torch.float32)
                tensor_spg = torch.tensor(spg, dtype=torch.int64)
                tensor_crysystem = torch.tensor(crysystem, dtype=torch.int64)
                if self.encode_element:
                    return {
                        # 'latt_dis': tensor_latt_dis,
                        'intensity': tensor_intensity,
                        'spg': tensor_spg,
                        'crysystem': tensor_crysystem,
                        'element': element_value
                    }
                else:
                    return {
                        # 'latt_dis': tensor_latt_dis,
                        'intensity': tensor_intensity,
                        'spg': tensor_spg,
                        'crysystem': tensor_crysystem,
                        'element': torch.tensor([])
                    }              
            cumulative_length += len(db)

    def symbol_to_atomic_number(self,symbol_list):
        # Mapping of element symbols to atomic numbers
        atomic_numbers = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5,
            'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
            'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
            'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
            'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35,
            'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
            'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45,
            'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55,
            'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
            'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65,
            'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
            'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75,
            'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
            'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
            'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
            'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95,
            'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
            'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
            'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
            'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115,
            'Lv': 116, 'Ts': 117, 'Og': 118
        }
        
        atomic_number_list = []
        for symbol in symbol_list:
            if symbol in atomic_numbers:
                atomic_number_list.append(atomic_numbers[symbol])
            else:
                atomic_number_list.append(0)  # Append None if symbol not in the dictionary
        
        return atomic_number_list
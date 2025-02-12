
# SimXRD-4M
## [HomePage](https://github.com/Bin-Cao/SimXRD)

## Installation

```
conda create -n xrdbench python=3.10
conda activate xrdbench
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Run the Project
```
bash sh/run.sh
```

## acquire the review data by Croissant

+ [review data](https://huggingface.co/datasets/caobin/SimXRDreview)


``` javascript
# 1. Point to the Croissant file
    import mlcroissant as mlc
    url = "https://huggingface.co/datasets/caobin/SimXRDreview/raw/main/simxrd_croissant.json"

# 2. Inspect metadata
  dataset_info = mlc.Dataset(url).metadata.to_json
  print(dataset_info)

  from dataset.parse import load_dataset,bar_progress # defined in our github : https://github.com/compasszzn/XRDBench/blob/main/dataset/parse.py
  for file_info in dataset_info['distribution']:
      wget.download(file_info['contentUrl'], './', bar=bar_progress)

# 3. Use Croissant dataset in your ML workload
  train_loader = DataLoader(load_dataset(name='train.tfrecord'), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
  val_loader = DataLoader(load_dataset(name='val.tfrecord'), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,drop_last=False)
  test_loader = DataLoader(load_dataset(name='test.tfrecord'), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,drop_last=False)
```


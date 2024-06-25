
# SimXRD-4M
## Benchmark ｜ [Database](https://github.com/Bin-Cao/SimXRD)
**Open Source:**  SimXRD-4M is freely available on our website (http://simxrd.caobin.asia/).

## Paper 
+ [arxiv](https://arxiv.org/pdf/2406.15469v1)
  
## Installation

To get started with XRDBench, you'll need to install the following libraries:

- PyTorch
- tqdm
- Weights & Biases (wandb)
- reformer-pytorch
- einops

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

## Kaggle Competition Announcement

To benchmark advanced models and further their development, we are launching a Kaggle competition for space group classification. Participants are invited to upload their predictions based on the [testNOtgt data](https://github.com/Bin-Cao/SimXRD/tree/main/testNOtgt_db) using their trained models. Submit your results on the [Leaderboard](https://www.kaggle.com/competitions/simxrd/leaderboard). 

For more detailed information, please visit the [Kaggle competition page](https://www.kaggle.com/competitions/simxrd).

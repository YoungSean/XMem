from dataset.static_dataset import StaticTransformDataset
from os import path
from torch.utils.data import DataLoader, ConcatDataset
import torch.distributed as distributed


static_root = './static'


# def construct_loader(dataset):
#     train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=local_rank, shuffle=True)
#     train_loader = DataLoader(dataset, config['batch_size'], sampler=train_sampler, num_workers=config['num_workers'],
#                               worker_init_fn=worker_init_fn, drop_last=True)
#     return train_sampler, train_loader


train_dataset = StaticTransformDataset(
            [
                (path.join(static_root, 'fss'), 0, 1),
                (path.join(static_root, 'DUTS-TR'), 1, 1),
                (path.join(static_root, 'DUTS-TE'), 1, 1),
                (path.join(static_root, 'ecssd'), 1, 1),
                (path.join(static_root, 'BIG_small'), 1, 5),
                (path.join(static_root, 'HRSOD_small'), 1, 5),
            ], num_frames=3)
        # train_sampler, train_loader = construct_loader(train_dataset)

print(train_dataset)
first_sample= train_dataset[0]
print(first_sample['info'])
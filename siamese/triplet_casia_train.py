""" The optimization of triplet training is highly dependent on the environment.
Especially the memory available on the GPU and workstation.

This script is merely meant to serve as a reference for the implementation of siamese triplet training.

Some of the questions to consider when implementing your own optimized script for triplet training are.
- Do I want to store all embeddings of a group on GPU or must I do so on RAM?
- Do I have the resources to compute the embeddings on the CPU with the help of an additional thread?
- Do I want to update weights by group or by batch
- Does it make sense to initialize workeres to do fetching of data

It can however also be used for training although the training will not be optimized,
70% average GPU utilization on a workstation with the following specs
Xeon e5 1620 v0
16gb RAM ddr3
SSD 256GB
GTX 1060 6GB
"""

import os
import logging

import torch

from torch_datasets.datasets.convert import convert_webface_to_siamese_dataset
from torch_datasets.utils.misc import cuda_batch

from torch_datasets import SiameseDataset, ImageCollateContainer, BalancedBatchSampler
from torch_collections import Siamese, negative_mining


# User defined locations
CASIA_ROOT_DIR = None
DATASET_CACHE = None
DEVICE_IDX = None
TOTAL_STEPS = 500000
STEPS_PER_EPOCH = 2500

BATCH_SIZE = 60
TRIPLET_BATCH_SIZE = BATCH_SIZE // 3

GROUP_SIZE           = 1024
NUM_PERSON_PER_GROUP = 20
NUM_IMG_PER_PERSON   = 40


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def make_folders():
    makedirs('logs')
    makedirs('snapshot')


def set_default_logging(log_path=None):
    """ Configs logging to the following settings
    - level set to INFO
    - logs saved to file and output to stdout
    - format in log file has the heading %(asctime)s [%(levelname)-4.4s]
    Args
        log_path : path to log file
    """
    # Make log path an abs path
    log_path = os.path.abspath(log_path)

    # Log to file
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='%(asctime)s [%(levelname)-4.4s] %(message)s',
        datefmt='%m-%d %H:%M',
        level=logging.INFO
    )

    # Log to stdout
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info('logging will be automatically saved to {}'.format(log_path))


def load_dataset():
    """ Dynamically loads dataset from cache or makes dataset from source """
    if DATASET_CACHE is not None:
        if os.path.isfile(DATASET_CACHE):
            dataset = SiameseDataset(DATASET_CACHE)
            logging.info('Dataset loaded from cache')
        else:
            dataset = convert_webface_to_siamese_dataset(CASIA_ROOT_DIR)
            dataset.save_dataset(DATASET_CACHE)
            logging.info('Dataset loaded and cached')
    else:
        dataset = convert_webface_to_siamese_dataset(CASIA_ROOT_DIR)
        logging.info('Dataset loaded')

    return dataset


def load_model():
    """ Builds/load a model appropriate for the dataset """
    encoder = Siamese(input_size=[160, 160]).train()
    if torch.cuda.is_available() and DEVICE_IDX is not None:
        encoder = encoder.cuda(DEVICE_IDX)
    logging.info('Model loaded')
    return retinanet


def encode_batch(image_ids, encoder, dataset, collate_container, batch_size=60):
    embeddings = []
    labels     = []
    for i in range(0, len(image_ids), batch_size):
        # Get image batch
        batch_ids = image_ids[i:i+batch_size]
        batch = [dataset[i] for i in batch_ids]
        image_batch = collate_container.collate_fn(batch)
        if torch.cuda.is_available() and DEVICE_IDX is not None:
            image_batch = image_batch.cuda(DEVICE_IDX)

        # Compute embeddings
        embeddings_batch = encoder(image_batch).cpu()

        # Store embeddings and labels
        embeddings = embeddings + [e for e in embeddings_batch]
        labels     = labels + [b['label'] for b in batch]

    # Compile embeddings and labels
    embeddings = torch.stack(embeddings, dim=0)
    labels     = torch.LongTensor(labels)

    return embeddings, labels


def main():
    # Initialize logging and folders
    make_folders()
    set_default_logging('logs/train.log')

    # Load dataset
    dataset = load_dataset()

    # Load model
    encoder = Siamese(input_size=[160, 160])
    logging.info('Model loaded')

    # Load dataset helpers
    collate_container = ImageCollateContainer(
        image_width=160,
        image_height=160,
        allow_transform=True
    )
    sampler = BalancedBatchSampler(
        dataset.get_all_image_class_ids(),
        batch_size=GROUP_SIZE,
        steps=TOTAL_STEPS,  # Use an arbitrary large step size
        n_classes=NUM_PERSON_PER_GROUP,
        n_samples=NUM_IMG_PER_PERSON
    )
    triplet_selector = negative_mining.HardestNegativeTripletSelector(encoder.configs['margin'], cpu=True)

    # Initialize optimizer and training variables
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=0.00001)
    done = False  # done acts as a loop breaker
    count_steps = 0
    cum_loss = 0
    cum_triplets_per_batch = 0
    logging.info('Starting training')

    while not done:
        for group_ids in sampler:
            # Get group embeddings
            group_embeddings, group_labels = encode_batch(group_ids, encoder, dataset, collate_container, batch_size=BATCH_SIZE)

            # Get triplets based on negative mining strategy
            triplet_ids = triplet_selector.get_triplets(group_embeddings, group_labels)
            triplet_ids.apply_(lambda ii: group_ids[ii])

            # Train model on triplets
            for i in range(0, len(triplet_ids), TRIPLET_BATCH_SIZE):
                # Load image batch
                batch_ids = triplet_ids[i:i+TRIPLET_BATCH_SIZE].transpose(0, 1).reshape(-1)
                batch = [dataset[i] for i in batch_ids]
                image_batch = collate_container.collate_fn(batch)
                if torch.cuda.is_available() and DEVICE_IDX is not None:
                    image_batch = image_batch.cuda(DEVICE_IDX)

                # zero optimizer
                optimizer.zero_grad()

                # Compute embeddings
                embeddings = encoder(image_batch)

                anchor   = embeddings[TRIPLET_BATCH_SIZE*0:TRIPLET_BATCH_SIZE*1]
                positive = embeddings[TRIPLET_BATCH_SIZE*1:TRIPLET_BATCH_SIZE*2]
                negative = embeddings[TRIPLET_BATCH_SIZE*2:TRIPLET_BATCH_SIZE*3]

                loss = encoder.triplet_loss(anchor, positive, negative)

                # backward + optimize
                loss.backward()
                optimizer.step()

                # Update training statistics
                count_steps += 1
                cum_loss = 0.99 * cum_loss + loss.detach()

                if count_steps % STEPS_PER_EPOCH == 0:
                    # Record weights as latest
                    torch.save(retinanet, 'snapshot/epoch_latest.pth')

                    # print current epoch information
                    cur_epoch = int(count_steps / STEPS_PER_EPOCH)
                    logging.info('Epoch: {} - triplet/batch: {:.1f} - loss: {:.5f}'.format(
                        cur_epoch,
                        cum_triplets_per_batch / 5,
                        cum_loss / 100
                    ))

            # Update training statistics
            cum_triplets_per_batch = 0.8 * cum_triplets_per_batch + len(triplet_ids)

            if count_steps >= TOTAL_STEPS:
                # Stop loop when required number of steps are passed
                done = True
                break

    torch.save(encoder, 'snapshot/epoch_final.pth')
    logging.info('Finished Training')


if __name__ == '__main__':
    # This step became required for python 3.X to share tensors for multiprocessing
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()

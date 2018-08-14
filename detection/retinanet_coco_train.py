"""
Simple script used to train the RetinaNet on the coco dataset on a single GPU
Evaluations are not conducted in this script, you should run eval using a separate thread to not slow down training
"""
import os
import logging

import torch

from torch_datasets.datasets.convert import convert_coco_to_detection_dataset
from torch_datasets import DetectionDataset, DetectionCollateContainer
from torch_collections import RetinaNet


# User defined locations
COCO_ANN_FILE = None
ROOT_IMAGE_DIR = None
DATASET_CACHE = None
DEVICE_IDX = 0
TOTAL_STEPS = 500000
STEPS_PER_EPOCH = 5000


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


def main():
    # Initialize logging and folders
    make_folders()
    set_default_logging('logs/train.log')

    # Load dataset
    if DATASET_CACHE is not None:
        if os.path.isfile(DATASET_CACHE):
            dataset = DetectionDataset(DATASET_CACHE)
            logging.info('Dataset loaded from cache')
        else:
            dataset = convert_coco_to_detection_dataset(
                COCO_ANN_FILE,
                ROOT_IMAGE_DIR,
                no_crowd=True
            )
            dataset.save_dataset(DATASET_CACHE)
            logging.info('Dataset loaded and cached')
    else:
        dataset = convert_coco_to_detection_dataset(
            COCO_ANN_FILE,
            ROOT_IMAGE_DIR,
            no_crowd=True
        )
        logging.info('Dataset loaded')

    # Create data loader
    collate_container = DetectionCollateContainer(allow_transform=True)
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_container.collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    retinanet = RetinaNet(num_classes=dataset.get_num_classes()).train().cuda(DEVICE_IDX)
    logging.info('Model loaded')

    # Initialize optimizer and training variables
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, retinanet.parameters()), lr=0.00001)

    done = False  # done acts as a loop breaker
    count_steps = 0
    cum_loss = 0
    best_loss = 1e5
    logging.info('Starting training')

    while not done:
        for batch in dataset_loader:
            # Copy batch to GPU
            image = batch['image'].cuda(DEVICE_IDX)
            annotations = [ann.cuda(DEVICE_IDX) for ann in batch['annotations']]

            # zero optimizer
            optimizer.zero_grad()

            # forward
            loss = retinanet(image, annotations)

            # In the event that loss is 0, None will be returned instead
            # This acts as a flag to signify that step can be skipped to
            # save the effort of backproping
            if loss is None:
                continue

            # backward + optimize
            loss.backward()
            optimizer.step()

            # Update training statistics
            count_steps += 1
            cum_loss = 0.8 * cum_loss + loss.detach()

            if count_steps % STEPS_PER_EPOCH == 0:
                # Record weights as latest
                torch.save(retinanet, 'snapshot/epoch_latest.pth')

                # Record best epoch
                if cum_loss < best_loss:
                    best_loss = cum_loss
                    torch.save(retinanet, 'snapshot/epoch_best.pth')

                # print current epoch information
                cur_epoch = int(count_steps / STEPS_PER_EPOCH)
                logging.info('Epoch: {} - loss: {}'.format(cur_epoch, cum_loss / 5))

            if count_steps >= TOTAL_STEPS:
                # Stop loop when required number of steps are passed
                done = True
                break

    torch.save(retinanet, 'snapshot/epoch_final.pth')
    logging.info('Finished Training')


if __name__ == '__main__':
    # This step became required for python 3.X to share tensors for multiprocessing
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()

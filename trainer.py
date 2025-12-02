import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F


from utils import DiceLoss


def compute_metrics(outputs, labels, num_classes):
    """
    outputs: (B, C, H, W)
    labels:  (B, H, W)
    """
    preds = torch.argmax(outputs, dim=1) 

    correct = (preds == labels).sum().item()
    total = labels.numel()
    pixel_acc = correct / total

    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        label_inds = (labels == cls)

        intersection = (pred_inds & label_inds).sum().item()
        union = (pred_inds | label_inds).sum().item()

        if union == 0:
            iou = float("nan")
        else:
            iou = intersection / union
        ious.append(iou)

    return pixel_acc, ious



def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    pred_dir = os.path.join(snapshot_path, "val_predictions")
    os.makedirs(pred_dir, exist_ok=True)

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    db_train = Synapse_dataset(
    base_dir=args.root_path,
    list_dir=args.list_dir,
    split="train",
    transform=transforms.Compose([
        RandomGenerator(output_size=args.img_size)
    ])
)

    db_val = Synapse_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="val",
        transform=transforms.Compose([
            RandomGenerator(output_size=args.img_size)
        ])
    )

    print("The length of train set is: {}".format(len(db_train)))

    

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader) 
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    best_loss = 10e10
    for epoch_num in iterator:
        model.train()
        batch_dice_loss = 0
        batch_ce_loss = 0
        epoch_train_acc = 0
        for i_batch, sampled_batch in tqdm(enumerate(train_loader), desc=f"Train: {epoch_num}", total=len(train_loader), leave=False):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            with torch.no_grad():
                train_pixel_acc, _ = compute_metrics(outputs, label_batch, num_classes)
            epoch_train_acc += train_pixel_acc

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            batch_dice_loss += loss_dice.item()
            batch_ce_loss += loss_ce.item()
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        batch_ce_loss /= len(train_loader)
        batch_dice_loss /= len(train_loader)
        batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
        epoch_train_acc /= len(train_loader) 
        logging.info('Train epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f, train_acc: %.4f' %
             (epoch_num, batch_loss, batch_ce_loss, batch_dice_loss, epoch_train_acc))
        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            batch_dice_loss = 0
            batch_ce_loss = 0
            with torch.no_grad():
                for i_batch, sampled_batch in tqdm(enumerate(val_loader), desc=f"Val: {epoch_num}",
                                                   total=len(val_loader), leave=False):
                    image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                    outputs = model(image_batch)

                    with torch.no_grad():
                        preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)  
                        preds_np = preds.cpu().numpy()

                        for b in range(preds_np.shape[0]):
                            mask = preds_np[b].astype("uint8")

                        

                            case_name = sampled_batch["case_name"][b] 

                            save_path = os.path.join(
                                pred_dir,
                                f"{case_name}_epoch{epoch_num}.png"
                            )

                            from PIL import Image
                            Image.fromarray(mask).save(save_path)

                    loss_ce = ce_loss(outputs, label_batch[:].long())
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    batch_dice_loss += loss_dice.item()
                    batch_ce_loss += loss_ce.item()
                    pixel_acc, ious = compute_metrics(outputs, label_batch, num_classes)

                    if i_batch == 0:
                        total_acc = pixel_acc
                        total_ious = np.array(ious)
                    else:
                        total_acc += pixel_acc
                        total_ious += np.array(ious)

                batch_ce_loss /= len(val_loader)
                batch_dice_loss /= len(val_loader)
                batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
                logging.info('Val epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
                    epoch_num, batch_loss, batch_ce_loss, batch_dice_loss))
                avg_acc = total_acc / len(val_loader)
                avg_ious = total_ious / len(val_loader)

                logging.info(f"Val epoch {epoch_num}: Pixel Accuracy = {avg_acc:.4f}")
                for cls_id, iou in enumerate(avg_ious):
                    logging.info(f"  Class {cls_id} IoU = {iou:.4f}")
                if batch_loss < best_loss:
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    best_loss = batch_loss
                else:
                    save_mode_path = os.path.join(snapshot_path, 'last_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return "Training Finished!"

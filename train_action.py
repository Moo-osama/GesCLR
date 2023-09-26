import argparse
import csv
import errno
import os
import random
import sys
import time
from collections import OrderedDict

import numpy as np
import tensorboardX
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mmcv import Config
from mmcv.runner import get_dist_info
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.data.dataset_action import NTURGBD
from lib.model.loss import *
from lib.model.model_action import ActionNet
from lib.utils.learning import *
from lib.utils.tools import *
from pyskl.datasets import build_dataloader, build_dataset
from pyskl.utils import mc_off, mc_on, test_port

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

"""

python train_action.py \
    --config configs/action/MB_train_k400_train_atten.yaml \
    --checkpoint exps/k400 \
    --config_pyskl configs/posec3d/slowonly_r50_346_k400/joint.py \
    --pretrained /home/osabdelfattah/TCL/mb_pretrained/mb_pretrained_light.bin

"""
def log_stuff(log_file, log):
    print (log)
    #if (not is_evaluate):
    with open(log_file, "a") as myfile:
        myfile.write(log + "\n")
        myfile.flush()
    myfile.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default="checkpoint",
        type=str,
        metavar="PATH",
        help="checkpoint directory",
    )
    parser.add_argument(
        "-p",
        "--pretrained",
        default="checkpoint",
        type=str,
        metavar="PATH",
        help="pretrained checkpoint directory",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default="",
        type=str,
        metavar="FILENAME",
        help="checkpoint to resume (file name)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        default="",
        type=str,
        metavar="FILENAME",
        help="checkpoint to evaluate (file name)",
    )
    parser.add_argument("-freq", "--print_freq", default=100)
    parser.add_argument(
        "-ms",
        "--selection",
        default="latest_epoch.bin",
        type=str,
        metavar="FILENAME",
        help="checkpoint to finetune (file name)",
    )
    parser.add_argument(
        "--config_pyskl", type=str, default="", help="train config file path"
    )
    opts = parser.parse_args()
    return opts


def validate(test_loader, model, criterion, last=False):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        end = time.time()
        # for idx, (batch_input, batch_gt) in tqdm(enumerate(test_loader)):  # (N, 2, T, 17, 3)
        class_accs = {}
        video = {}

        for idx, batch in tqdm(enumerate(test_loader)):
            batch_gt = batch["label"]
            batch_input = batch["keypoint"].float()

            batch_size = len(batch_input)
            if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input = batch_input.cuda()
            output = model(batch_input)  # (N, num_classes)

            loss = criterion(output, batch_gt.squeeze())

            preds = output.argmax(dim=1)
            accs = (preds == batch_gt.squeeze()).float()

            for i in range(batch_size):
                video[batch_gt[i]] = accs[i].item()
                class_idx = batch_gt[i]
                if class_idx.item() not in class_accs:
                    class_accs[class_idx.item()] = [
                        0,
                        0,
                    ]  # [accumulated accuracy, number of samples]
                class_accs[class_idx.item()][0] += accs[i].item()
                class_accs[class_idx.item()][1] += 1

            # update metric
            losses.update(loss.item(), batch_size)
            acc1, acc5 = accuracy(output, batch_gt, topk=(1, 5))
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % opts.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t".format(
                        idx,
                        len(test_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                        top5=top5,
                    )
                )
        if last:
            class_avg_accs = {}
            for class_idx, (accum_acc, num_samples) in class_accs.items():
                avg_acc = accum_acc / num_samples
                class_name = f"class_{class_idx}"
                class_avg_accs[class_name] = avg_acc
            num_person = batch_input.shape[1]
            txt = open("num_person_MB.txt", "r")
            real_num_person = txt.read().split(",")
            real_num_person = real_num_person[:-1]
            real_num_person = [int(x) for x in real_num_person]
            cap_num_person = [10 if x > 10 else x for x in real_num_person]
            with open(
                "class_avg_accs" + str(num_person) + ".csv", "w", newline=""
            ) as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["class_name", "avg_top1_accuracy"])
                for class_name, avg_acc in class_avg_accs.items():
                    writer.writerow([class_name, avg_acc])
            with open(
                "video_avg_accs" + str(num_person) + ".csv", "w", newline=""
            ) as csvfile2:
                writer = csv.writer(csvfile2)
                writer.writerow(["num_person", "class_name", "top1_accuracy"])
                for num, (label, vacc) in zip(cap_num_person, video.items()):
                    writer.writerow([num, label.item(), vacc])

        return losses.avg, top1.avg, top5.avg


def train_with_config(args, opts):

    log_file_name = opts.checkpoint + "/logs"+ ".txt"

    with open("num_person_MB.txt", "w") as file:
        file.write("")
    print(args)
    cfg = Config.fromfile(opts.config_pyskl)
    rank, _ = get_dist_info()

    default_mc_cfg = ("localhost", 22077)
    memcached = cfg.get("memcached", False)

    if rank == 0 and memcached:
        # mc_list is a list of pickle files you want to cache in memory.
        # Basically, each pickle file is a dictionary.
        mc_cfg = cfg.get("mc_cfg", default_mc_cfg)
        assert isinstance(mc_cfg, tuple) and mc_cfg[0] == "localhost"
        if not test_port(mc_cfg[0], mc_cfg[1]):
            mc_on(port=mc_cfg[1], launcher="pytorch")
        retry = 3
        while not test_port(mc_cfg[0], mc_cfg[1]) and retry > 0:
            time.sleep(5)
            retry -= 1
        assert retry >= 0, "Failed to launch memcached. "

    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError(
                "Unable to create checkpoint directory:", opts.checkpoint
            )
    # train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))
    model_backbone = load_backbone(args)
    model_backbone = model_backbone.float()
    if args.finetune:
        if opts.resume or opts.evaluate:
            pass
        else:
            chk_filename = opts.pretrained #os.path.join(opts.pretrained, opts.selection)
            print("Loading backbone", chk_filename)
            checkpoint = torch.load(
                chk_filename, map_location=lambda storage, loc: storage
            )["model_pos"]
            model_backbone = load_pretrained_weights(model_backbone, checkpoint)
    if args.partial_train:
        model_backbone = partial_train_layers(model_backbone, args.partial_train)

    # uncomment to freeze backbone

    # for param in model_backbone.parameters():
    #     param.requires_grad = False

    model = ActionNet(
        backbone=model_backbone,
        dim_rep=args.dim_rep,
        num_classes=args.action_classes,
        dropout_ratio=args.dropout_ratio,
        version=args.model_version,
        hidden_dim=args.hidden_dim,
        with_attention=args.with_attention,
    )
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
    best_acc = 0
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print("INFO: Trainable parameter count:", model_params)
    print("Loading dataset...")
    trainloader_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": 8,
        "pin_memory": True,
        "prefetch_factor": 4,
        "persistent_workers": True,
    }

    #train_dataset = build_dataset(cfg.data.train)
    #train_loader = DataLoader(train_dataset, **trainloader_params, drop_last=True)
    test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    train_loader = test_loader = DataLoader(test_dataset, **trainloader_params, drop_last=True)

    #chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
    #if os.path.exists(chk_filename):
    #    opts.resume = chk_filename
    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        print("Loading checkpoint", chk_filename)
        if os.path.exists(chk_filename):
            checkpoint = torch.load(
                chk_filename, map_location=lambda storage, loc: storage
            )
            model.load_state_dict(checkpoint["model"], strict=True)

    if not opts.evaluate:
        optimizer = optim.AdamW(
            [
                {
                    "params": filter(
                        lambda p: p.requires_grad, model.module.backbone.parameters()
                    ),
                    "lr": args.lr_backbone,
                },
                {
                    "params": filter(
                        lambda p: p.requires_grad, model.module.head.parameters()
                    ),
                    "lr": args.lr_head,
                },
            ],
            lr=args.lr_backbone,
            weight_decay=args.weight_decay,
        )

        scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)
        st = 0
        print("INFO: Training on {} batches".format(len(train_loader)))
        if opts.resume:
            st = checkpoint["epoch"]
            if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            else:
                print(
                    "WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized."
                )
            lr = checkpoint["lr"]
            if "best_acc" in checkpoint and checkpoint["best_acc"] is not None:
                best_acc = checkpoint["best_acc"]
        # Training
        for epoch in range(st, args.epochs):
            print("Training epoch %d." % epoch)
            losses_train = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            model.train()
            end = time.time()
            iters = len(train_loader)
            step = 0
            for idx, batch in tqdm(enumerate(train_loader)):  # (N, 2, T, 17, 3)
                step += 1

                batch_gt = batch["label"]
                batch_input = batch["keypoint"].float()
                if step == 1:
                    print("batch input size ", batch_input.shape)

                if batch_input.shape[0] == 1:
                    print("SHAPE IS WRONG !!!!", flush=True)
                    print("batch input size ", batch_input.shape, flush=True)
                data_time.update(time.time() - end)
                batch_size = len(batch_input)
                if torch.cuda.is_available():
                    batch_gt = batch_gt.cuda()
                    batch_input = batch_input.cuda()
                model = model.float()
                output = model(batch_input)  # (N, num_classes)
                optimizer.zero_grad()
                loss_train = criterion(output, batch_gt.squeeze())
                losses_train.update(loss_train.item(), batch_size)
                acc1, acc5 = accuracy(output, batch_gt, topk=(1, 5))
                top1.update(acc1[0], batch_size)
                top5.update(acc5[0], batch_size)
                loss_train.backward()
                optimizer.step()
                batch_time.update(time.time() - end)
                end = time.time()
                if (idx + 1) % opts.print_freq == 0:
                    log = str(
                        "Train: [{0}][{1}/{2}]\t"
                        "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                        "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                        "Acc@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                            epoch,
                            idx + 1,
                            len(train_loader),
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses_train,
                            top1=top1,
                        )
                    )

                    log_stuff(log_file_name, log)
                    sys.stdout.flush()

                torch.cuda.empty_cache()

            scheduler.step()

            # Save latest checkpoint.
            chk_path = os.path.join(opts.checkpoint, "latest_epoch.bin")
            log = "Saving checkpoint to" + str(chk_path)

            log_stuff(log_file_name, log)
            sys.stdout.flush()

            torch.save(
                {
                    "epoch": epoch + 1,
                    "lr": scheduler.get_last_lr(),
                    "optimizer": optimizer.state_dict(),
                    "model": model.state_dict(),
                    "best_acc": best_acc,
                },
                chk_path,
            )

            # if opts.evaluate:
            test_loss, test_top1, test_top5 = losses_train.avg, top1.avg, top5.avg

            if (idx + 1) % 10 == 0:
                test_loss, test_top1, test_top5 = validate(
                    test_loader, model, criterion, last=False
                )

            log = str(
                "epoch {ep} \t"
                "Loss {loss:.4f} \t"
                "Acc@1 {top1:.3f} \t"
                "Acc@5 {top5:.3f} \t".format(
                    ep=epoch, loss=test_loss, top1=test_top1, top5=test_top5
                )
            )

            log_stuff(log_file_name, log)
            sys.stdout.flush()


            acc_name = ""
            print("test_top1.item(): ", test_top1.item())
            if test_top1.item() > 40:
                acc_name = '_' + str(round(test_top1.item(),2))
            
                best_chk_path = os.path.join(opts.checkpoint, 'best_epoch%s.bin' % (acc_name))

                torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
                }, best_chk_path)

        test_loss, test_top1, test_top5 = validate(
            test_loader, model, criterion, last=True
        )
        log = str (
            "Loss {loss:.4f} \t"
            "Acc@1 {top1:.3f} \t"
            "Acc@5 {top5:.3f} \t".format(loss=test_loss, top1=test_top1, top5=test_top5)
        )

        log_stuff(log_file_name, log)
        sys.stdout.flush()

    if rank == 0 and memcached:
        mc_off()


if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    train_with_config(args, opts)

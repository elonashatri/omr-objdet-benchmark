import os
import json
import argparse
import time
import datetime
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
from torchvision.models import resnet50
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as T
from tqdm import tqdm

# Import the Doremi dataset class
from detr_doremi_loader import DoremiDataset

# DETR components
class DETR(nn.Module):
    """
    End-to-End Object Detection with Transformers
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        
        # Create ResNet50 backbone
        self.backbone = resnet50(pretrained=True)
        del self.backbone.fc
        
        # Create projection layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        
        # Create transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers
        )
        
        # Prediction heads for class and bounding box
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.linear_bbox = nn.Linear(hidden_dim, 4)  # (x1, y1, x2, y2)
        
        # Object queries - these are learned parameters
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        
        # Output positional encodings (object queries)
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        
        # Reset parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, inputs):
        # Extract features from backbone
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Projection
        h = self.conv(x)
        
        # Positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        
        # Transformer expects sequence first, batch second
        h = h.flatten(2).permute(2, 0, 1)
        query_pos = self.query_pos.unsqueeze(1).repeat(1, inputs.shape[0], 1)
        query = torch.zeros_like(query_pos)
        
        # Transformer
        memory = self.transformer.encoder(h, None, pos)  # Don't mask encoded sequence
        h = self.transformer.decoder(query, memory, None, None, query_pos, pos)
        
        # Prediction heads
        outputs_class = self.linear_class(h)
        outputs_coord = self.linear_bbox(h).sigmoid()  # Sigmoid to get [0, 1] range
        
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for DETR
    """
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        # Compute the classification cost
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Compute the GIoU cost between boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / union
    
    lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    
    whi = (rbi - lti).clamp(min=0)  # [N,M,2]
    areai = whi[:, :, 0] * whi[:, :, 1]
    
    return iou - (areai - union) / areai

# Linear sum assignment using scipy
def linear_sum_assignment(cost_matrix):
    try:
        import scipy.optimize
        return scipy.optimize.linear_sum_assignment(cost_matrix)
    except ImportError:
        raise ImportError('Please run pip install scipy to use Hungarian matching.')

class SetCriterion(nn.Module):
    """
    Loss function for DETR
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
    
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """L1 box regression loss"""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def forward(self, outputs, targets):
        """This performs the loss computation."""
        indices = self.matcher(outputs, targets)
        
        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if dist.is_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / dist.get_world_size() if dist.is_initialized() else num_boxes, min=1).item()
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        
        return losses
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)

def build_detr(args):
    """Build the DETR model with all components"""
    num_classes = args.num_classes
    device = torch.device(args.device)
    
    # Create DETR model
    model = DETR(
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
    )
    
    # Create matcher
    matcher = HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
    )
    
    # Create loss function
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
    losses = ['labels', 'boxes']
    
    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
    )
    
    model.to(device)
    criterion.to(device)
    
    return model, criterion

def collate_fn(batch):
    """Collate function for the DataLoader"""
    images = []
    targets = []
    
    for img, tgt in batch:
        # Convert numpy arrays to tensors if needed
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        images.append(img)
        
        # Prepare target dict for DETR format
        boxes = tgt["boxes"]
        
        # Convert box coordinates from [x1, y1, x2, y2] to [cx, cy, w, h] format
        boxes_cxcywh = box_xyxy_to_cxcywh(boxes)
        
        # Normalize coordinates to [0, 1]
        h, w = img.shape[-2:]
        boxes_cxcywh = boxes_cxcywh / torch.tensor([w, h, w, h], dtype=torch.float32)
        
        # Create new target dict
        new_target = {
            "boxes": boxes_cxcywh,
            "labels": tgt["labels"],
            "image_id": tgt["image_id"]
        }
        
        targets.append(new_target)
    
    return torch.stack(images), targets

class DoremiTransform:
    """Transforms for the Doremi dataset"""
    
    def __init__(self, image_size=800):
        self.image_size = image_size
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __call__(self, img, target):
        # Resize image and bounding boxes
        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        else:
            h, w = img.shape[-2:]
        
        # Apply transforms
        img = self.transforms(img)
        
        if "boxes" in target:
            boxes = target["boxes"]
            
            # Convert boxes to cxcywh format
            if len(boxes) > 0:
                boxes_cxcywh = box_xyxy_to_cxcywh(boxes)
                
                # Normalize coordinates
                boxes_cxcywh = boxes_cxcywh / torch.tensor([w, h, w, h], dtype=torch.float32)
                
                target["boxes"] = boxes_cxcywh
            else:
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        
        return img, target

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, writer=None, rank=0, print_freq=10):
    """Train the model for one epoch"""
    model.train()
    criterion.train()
    
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        outputs = model(torch.stack(images))
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # Reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        
        optimizer.zero_grad()
        losses.backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()
        
        metric_logger.update(loss=losses_reduced_scaled, **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    # Log to TensorBoard
    if writer is not None and rank == 0:
        writer.add_scalar('training/loss', metric_logger.meters['loss'].global_avg, epoch)
        writer.add_scalar('training/loss_ce', metric_logger.meters['loss_ce'].global_avg, epoch)
        writer.add_scalar('training/loss_bbox', metric_logger.meters['loss_bbox'].global_avg, epoch)
        writer.add_scalar('training/loss_giou', metric_logger.meters['loss_giou'].global_avg, epoch)
        writer.add_scalar('training/learning_rate', optimizer.param_groups[0]["lr"], epoch)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, criterion, data_loader, device, epoch, writer=None, rank=0):
    """Evaluate the model on the validation set"""
    model.eval()
    criterion.eval()
    
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        outputs = model(torch.stack(images))
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        # Reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        
        metric_logger.update(loss=losses_reduced_scaled, **loss_dict_reduced_scaled)
    
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    # Log to TensorBoard
    if writer is not None and rank == 0:
        writer.add_scalar('validation/loss', metric_logger.meters['loss'].global_avg, epoch)
        writer.add_scalar('validation/loss_ce', metric_logger.meters['loss_ce'].global_avg, epoch)
        writer.add_scalar('validation/loss_bbox', metric_logger.meters['loss_bbox'].global_avg, epoch)
        writer.add_scalar('validation/loss_giou', metric_logger.meters['loss_giou'].global_avg, epoch)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size <= 1:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = []
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
        self.window_size = window_size

    def update(self, value, n=1):
        self.deque.append(value)
        if len(self.deque) > self.window_size:
            self.deque.pop(0)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Synchronize values between processes (when using distributed training)
        """
        if not dist.is_available():
            return
        if not dist.is_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        self.count = int(t[0].item())
        self.total = t[1].item()

    @property
    def median(self):
        d = sorted(self.deque)
        return d[len(d) // 2] if d else 0.0

    @property
    def avg(self):
        d = self.deque
        return sum(d) / len(d) if d else 0.0

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def max(self):
        return max(self.deque) if self.deque else 0.0

    @property
    def value(self):
        return self.deque[-1] if self.deque else 0.0

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return object.__getattr__(self, attr)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        if header is not None:
            print(header)
        
        i = 0
        start_time = time.time()
        end = start_time
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        
        for obj in iterable:
            yield obj
            i += 1
            if i % print_freq == 0 or i == len(iterable):
                eta_seconds = (time.time() - start_time) / i * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                print(f"{header} [{i:{space_fmt}}/{len(iterable)}] " +
                      f"eta: {eta_string} " +
                      f"{str(self)}")

def setup(rank, world_size):
    """
    Initialize the distributed environment
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """
    Clean up the distributed environment
    """
    dist.destroy_process_group()

def train_model(rank, world_size, args):
    """
    Train the model on one GPU
    """
    # Initialize process group
    setup(rank, world_size)
    
    # Create output directory
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create tensorboard writer
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    else:
        writer = None
    
    # Set device
    device = torch.device(f"cuda:{rank}")
    
    # Load class mapping
    with open(args.class_mapping, 'r') as f:
        class_to_idx = json.load(f)
    
    args.num_classes = len(class_to_idx)
    if rank == 0:
        print(f"Using {args.num_classes} classes")
    
    # Build the model and criterion
    model, criterion = build_detr(args)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Define optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    
    # Define learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    # Create dataset and sampler for training
    train_dataset = DoremiDataset(
        args.data_dir,
        args.train_annotation_dir,
        args.class_mapping,
        min_box_size=args.min_box_size,
        max_classes=None,
        transform=DoremiTransform(image_size=args.img_size)
    )
    
    # Create distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Create validation dataset
    if args.val_annotation_dir:
        val_dataset = DoremiDataset(
            args.data_dir,
            args.val_annotation_dir,
            args.class_mapping,
            min_box_size=args.min_box_size,
            max_classes=None,
            transform=DoremiTransform(image_size=args.img_size)
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    else:
        val_dataloader = None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            if rank == 0:
                print(f"Loading checkpoint {args.resume}")
            
            # Map model to be loaded to specified single GPU
            loc = f'cuda:{rank}'
            checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint['epoch'] + 1
            model.module.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            
            if rank == 0:
                print(f"Loaded checkpoint {args.resume} (epoch {checkpoint['epoch']})")
        else:
            if rank == 0:
                print(f"No checkpoint found at {args.resume}")
    
    # Train the model
    if rank == 0:
        print("Start training")
    
    best_loss = float('inf')
    for epoch in range(start_epoch, args.epochs):
        if rank == 0:
            start_time = time.time()
        
        # Set the epoch for the train sampler
        train_sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_stats = train_one_epoch(model, criterion, train_dataloader, optimizer, device, epoch, writer, rank)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set
        if val_dataloader is not None:
            val_stats = evaluate(model, criterion, val_dataloader, device, epoch, writer, rank)
            val_loss = val_stats['loss']
            
            # Save best model
            if rank == 0 and val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'args': args,
                }, os.path.join(args.output_dir, f"best_model.pth"))
                print(f"New best model saved with loss: {val_loss:.4f}")
        
        # Save checkpoint
        if rank == 0 and (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
            }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth"))
            
            # Log training time
            if rank == 0:
                end_time = time.time()
                print(f"Epoch {epoch} completed in {end_time - start_time:.2f} seconds")
    
    # Save final model
    if rank == 0:
        torch.save({
            'epoch': args.epochs - 1,
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'args': args,
        }, os.path.join(args.output_dir, "final_model.pth"))
        
        if writer is not None:
            writer.close()
    
    # Clean up
    cleanup()

def main():
    parser = argparse.ArgumentParser(description="DETR for Doremi dataset")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--train_annotation_dir", type=str, required=True, help="Directory containing training XML annotations")
    parser.add_argument("--val_annotation_dir", type=str, default="", help="Directory containing validation XML annotations")
    parser.add_argument("--class_mapping", type=str, required=True, help="Path to class mapping JSON file")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of transformer")
    parser.add_argument("--nheads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="Number of decoder layers")
    
    # Loss parameters
    parser.add_argument("--set_cost_class", type=float, default=1, help="Class coefficient in the matching cost")
    parser.add_argument("--set_cost_bbox", type=float, default=5, help="L1 box coefficient in the matching cost")
    parser.add_argument("--set_cost_giou", type=float, default=2, help="GIoU box coefficient in the matching cost")
    parser.add_argument("--eos_coef", type=float, default=0.1, help="Relative classification weight of the 'no-object' class")
    parser.add_argument("--bbox_loss_coef", type=float, default=5, help="Box loss coefficient")
    parser.add_argument("--giou_loss_coef", type=float, default=2, help="GIoU loss coefficient")
    
    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for backbone")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--lr_drop", type=int, default=40, help="Learning rate drop step size")
    parser.add_argument("--clip_max_norm", type=float, default=0.1, help="Gradient clipping max norm")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    
    # Data processing parameters
    parser.add_argument("--min_box_size", type=int, default=5, help="Minimum size for bounding boxes")
    parser.add_argument("--img_size", type=int, default=800, help="Image size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # Device parameters
    parser.add_argument("--device", type=str, default="cuda:2", help="Device for training")
    parser.add_argument("--world_size", type=int, default=3, help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save command-line arguments
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Use all GPUs
    ngpus_per_node = min(torch.cuda.device_count(), args.world_size)
    print(f"Using {ngpus_per_node} GPUs")
    
    # Launch training processes
    mp.spawn(train_model, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

if __name__ == "__main__":
    main()
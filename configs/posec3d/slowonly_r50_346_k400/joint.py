data_prefix = "/data/os/kpfiles/"
memcached = True
mc_cfg = ("localhost", 22077)
dataset_type = "PoseDataset"
ann_file = "/home/osabdelfattah/MotionBert_k400/k400_hrnet.pkl"
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
box_thr = 0.5
valid_ratio = 0.0
persons = 6

train_pipeline = [
    dict(type="DecompressPose", squeeze=True, max_person=persons),
    dict(type="PoseDecode"),
    dict(type="ActionDataset", n_frames=75, is_train=True),
    dict(type="Collect", keys=["keypoint", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["keypoint", "label"], max_person=persons),
]
val_pipeline = [
    dict(type="DecompressPose", squeeze=True, max_person=persons),
    dict(type="PoseDecode"),
    dict(type="ActionDataset", n_frames=90, is_train=False),
    dict(type="Collect", keys=["keypoint", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["keypoint", "label"], max_person=persons),
]
test_pipeline = [
    dict(type="DecompressPose", squeeze=True, max_person=persons, isTest=True),
    dict(type="PoseDecode"),
    dict(type="ActionDataset", n_frames=90, is_train=False),
    dict(type="Collect", keys=["keypoint", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["keypoint", "label"], max_person=persons),
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=8,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file,
        split="train",
        pipeline=train_pipeline,
        box_thr=box_thr,
        valid_ratio=valid_ratio,
        memcached=memcached,
        mc_cfg=mc_cfg,
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file,
        split="val",
        pipeline=val_pipeline,
        box_thr=box_thr,
        memcached=memcached,
        mc_cfg=mc_cfg,
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file,
        split="val",
        pipeline=test_pipeline,
        box_thr=box_thr,
        memcached=memcached,
        mc_cfg=mc_cfg,
    ),
)

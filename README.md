# CS7303

Image Processing and Machine Vision

## Dataset

[ADE20K](https://ade20k.csail.mit.edu)

## Advanced Methods

### Mask2Former (Yushun Xiang)

> [Config file](Advanced/Mask2Former/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml)

Inference demo with pre-trained models:

```shell
cd Advanced/Mask2Former/demo/
python demo.py \
    --config-file ../configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml \
    --input input1.jpg input2.jpg \
    --opts MODEL.WEIGHTS models/model_final_6b4a3a.pkl
```

To evaluate a model's performance, use

```shell
cd Advanced/Mask2Former/
python train_net.py \
  --config-file configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml \
  --eval-only MODEL.WEIGHTS models/model_final_6b4a3a.pkl
```

For more options, see python train_net.py -h.

Results:

```log
[12/28 04:00:18 d2.engine.defaults]: Evaluation results for ade20k_sem_seg_val in csv format:
[12/28 04:00:18 d2.evaluation.testing]: copypaste: Task: sem_seg
[12/28 04:00:18 d2.evaluation.testing]: copypaste: mIoU,fwIoU,mACC,pACC
[12/28 04:00:18 d2.evaluation.testing]: copypaste: 56.0270,75.8601,69.3965,85.2141
```

### PointRend

> [Config file](Advanced/PointRend/config/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml)

Training and evaluate FCN + PointRend
```shell
cd Advanced/PointRend
python train.py
```

Results:
```log
```

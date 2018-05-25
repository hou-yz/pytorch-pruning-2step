# Train & Pruning with PyTorch
by hou-yz
Based on kuangliu/pytorch-cifar;


improve inference speed and reduce intermediate feature sizes to favor distributed inference (local device compute half of the model and upload the feature for further computing on stronger devices or cloud).

- pruning stage-1: prune the whole model to increase inference speed and slightly reduce intermediate feature sizes.

- pruning stage-2: for each split-point (where the intermediate feature is transferred to another device for further computation), specifically prune the layer just before the split-point to reduce intermediate feature sizes even more.

- model trained on cifar-10, tested only on vgg-16


## usage
- training:
```lua
python main.py --train          # train from scratch
python main.py --resume         # resume training
```

- 2-step pruning:
- first, in step-1, you can prune the whole model by
```lua 
python main.py --prune          # prune the whole model
```

- once you finished step-1, you can the prune each layer (step-2) individually form minimum bandwidth requirement with 
``` lua
python main.py --prune_layer    # prune layers and save models separately
```

- for excel chart drawing ang logging, try 
```lua
python draw_chart.py
```
which automatically generates the `chart.xlsx` file.


## updates
- added pruning features;
- added 2-stage pruning method: --prune & --prune_layer
- added draw_chart with `openpyxl` (in excel);
- added cpu-only support and windows support.

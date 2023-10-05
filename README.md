# Research Framework

### How to run

#### Single GPU

```bash
python main.py gpus=0
```

#### Multi GPU

```bash
torchrun --nproc_per_node=4 main.py gpus=[0,1,2,3]
```

#### Pre-defined setting

```bash
torchrun --nproc_per_node=2 main.py --config-name=a3_resnet gpus=[0,1]
```

#### Resume

```bash
python main.py train.resume=runs/modelName_dataset/exp_name
```

#### Sequential multi train

The resnet34 will be trained automatically after finishing resnet18 training.

```bash
torchrun --nproc_per_node=4 main.py gpus=[0,1,2,3] model.model_name=resnet18,resnet34,resnet50
```

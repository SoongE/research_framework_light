# Research Framework

##### How to install hydra

This project use [`hydra`](https://github.com/facebookresearch/hydra) library. So you need to install it. Please run
below code.

```bash
pip install hydra-core --upgrade
```

#### How to run

Single GPU

```bash
python main.py gpus=[0]
```

Multi GPU

```bash
torchrun --nproc_per_node=4 main.py gpus=[0,1,2,3]
```

Pre-defined setting

```bash
torchrun --nproc_per_node=4 main.py --config-name=a2_resnet wandb=True
```

Resume

```bash
python main.py train.resume=runs/modelName_dataset/exp_name
```
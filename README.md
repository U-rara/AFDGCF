## Requirements

```python
recbole>=1.0.0
```

## Examples

### Baseline

```python
python main.py --gpu_id=0 --log_wandb=True --model=ngcf --dataset=yelp-2018 --reg_weight=1e-5
python main.py --gpu_id=0 --log_wandb=True --model=gccf --dataset=yelp-2018 --reg_weight=1e-3
python main.py --gpu_id=0 --log_wandb=True --model=hmlet --dataset=yelp-2018
python main.py --gpu_id=0 --log_wandb=True --model=lightgcn --dataset=yelp-2018 --reg_weight=1e-4
```
### AFDGCF

```python
python main.py --gpu_id=0 --model=afd-ngcf --alpha=5e-4 --dataset=yelp-2018 --log_wandb=True
python main.py --gpu_id=0 --model=afd-gccf --alpha=5e-5 --dataset=yelp-2018 --log_wandb=True
python main.py --gpu_id=0 --model=afd-hmlet --alpha=1e-3 --dataset=yelp-2018 --log_wandb=True
python main.py --gpu_id=0 --model=afd-lightgcn --alpha=1e-4 --dataset=yelp-2018 --log_wandb=True

```



> The dataset files will be automatically downloaded, or you can access them at https://github.com/RUCAIBox/RecSysDatasets 
>
> Yelp-2018: https://drive.google.com/file/d/1KOnx0KKt69SCVA1H7E8vcc814tbNehTl/view?usp=drive_link ; Extract and place '/dataset/yelp-2018/yelp-2018.inter'
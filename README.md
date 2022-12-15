# Pix2Pix

My implementation of "Image-to-Image Translation with Conditional Adversarial Networks" paper

[Paper Pix2Pix](https://arxiv.org/pdf/1611.07004.pdf)

[All runs](https://wandb.ai/messlav/pix2pix)

[Wandb report](https://wandb.ai/messlav/pix2pix/reports/Pix2Pix--VmlldzozMTUxMDM3?accessToken=cw7eljlv7fi1wntvseghhmulmrrs09hw0jw3uxnvhpur35iwimn9gcy70brtf1rs)

# Reproduce code

1. Clone repo and install packages
```python
git clone https://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@github.com/messlav/pix2pix.git
cd pix2pix
pip install -r requirements.txt
```

2. Train generator on L1 loss
```python
python3 train_on_l1.py --dataset {facades/maps/flags}
```

3. Train GAN
```python
python3 train.py --dataset {facades/maps/flags}
```

4. Download weights (L1 facades and GAN maps)
```python
gdown 'https://drive.google.com/u/0/uc?id=1pHPJHY8Gri21t1gNpsCOBl-iGYeOHqW4'
gdown 'https://drive.google.com/u/0/uc?id=1_nz-Yfi4YyVBhsyi67g1b_bXSqXayC46'
```

5. Calculate fid by file with weights
```python
python3 fid_calculation.py --dataset {dataset} --G_path {path to file} 
```


# ORMamba: Ordinal Regressive Multiscale Temporal Fusion Mamba for Action Quality Assessment

This repository contains the official PyTorch implementation for ORMamba (**ICME 2026**). 

**⚠️ Project Status: The model checkpoints, and preprocessed datasets will be made publicly available upon the acceptance of our paper at ICME 2026. ⚠️**

## 🎯 Overview
we propose Ordinal Regression Mamba (*__ORMamba__*), an end-to-end AQA framework. ORMamba incorporates a multi-scale temporal fusion Mamba module to capture long-range temporal dependencies and explicitly models mistaken actions to learn more discriminative representations.Additionally, the framework integrates ordinal regression with contrastive regression for coarse-to-fine precise score estimation, which significantly improves the overall performance.

![intro](fig/ORMamba_overview.png)


## 📊 Main Results

<table style="width:100%; border-collapse:collapse; font-family:system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; font-size:14px;">
  <thead>
    <tr style="background:#d9e1f2;color:#0b3a8a;">
      <th rowspan="2" style="padding:10px 12px; text-align:center; border-right:1px solid #bbb;">Method</th>
      <th colspan="3" style="padding:10px 12px; text-align:center; border-right:1px solid #bbb;">FineDiving dataset</th>
      <th colspan="3" style="padding:10px 12px; text-align:center;">LOGO dataset</th>
    </tr>
    <tr style="background:#f5f7fb; color:#0b3a8a;">
      <th style="padding:8px 10px; text-align:center;">ρ ↑</th>
      <th style="padding:8px 10px; text-align:center;">R-ℓ₂ ↓</th>
      <th style="padding:8px 10px; text-align:center;">Publisher</th>
      <th style="padding:8px 10px; text-align:center;">ρ ↑</th>
      <th style="padding:8px 10px; text-align:center;">R-ℓ₂ ↓</th>
      <th style="padding:8px 10px; text-align:center;">Publisher</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#fff;">
      <td style="padding:10px 12px; text-align:center; border-top:1px solid #eee;">Previous SOTA</td>
      <td style="padding:10px 12px; text-align:center; border-top:1px solid #eee;">0.9501</td>
      <td style="padding:10px 12px; text-align:center; border-top:1px solid #eee;">0.2294</td>
      <td style="padding:10px 12px; text-align:center; border-top:1px solid #eee;">TPAMI 2025</td>
      <td style="padding:10px 12px; text-align:center; border-top:1px solid #eee;">0.7130</td>
      <td style="padding:10px 12px; text-align:center; border-top:1px solid #eee;">3.6080</td>
      <td style="padding:10px 12px; text-align:center; border-top:1px solid #eee;">TIP 2025</td>
    </tr>
    <tr style="background:#f0f4ff; font-weight:600;">
      <td style="padding:10px 12px; text-align:center; border-top:1px solid #eee;">ORMamba(Ours)</td>
      <td style="padding:10px 12px; text-align:center; border-top:1px solid #eee; color:#0b3a8a;">0.9505</td>
      <td style="padding:10px 12px; text-align:center; border-top:1px solid #eee; color:#0b3a8a;">0.2056</td>
      <td style="padding:10px 12px; text-align:center; border-top:1px solid #eee;">ICME 2026</td>
      <td style="padding:10px 12px; text-align:center; border-top:1px solid #eee; color:#0b3a8a;">0.7503</td>
      <td style="padding:10px 12px; text-align:center; border-top:1px solid #eee; color:#0b3a8a;">3.3357</td>
      <td style="padding:10px 12px; text-align:center; border-top:1px solid #eee;">ICME 2026</td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td colspan="7" style="padding:8px 12px; font-size:12px; color:#555;">
        Notes: R-ℓ₂ values are multiplied by 100. Arrows indicate metric direction (higher ρ better; lower R-ℓ₂ better).
      </td>
    </tr>
  </tfoot>
</table>



## 🚀 Usage

### 🛠️ Requirements

Make sure the following dependencies installed (python):

* pytorch >= 2.4.1
* einops
* timm
* torch_videovision
* CUDA >= 12.4
* triton == 3.1.0
* causal-conv1d
* mamba-ssm == 2.2.2



```
pip install git+https://github.com/hassony2/torch_videovision

cd mamba-requirements/causal-conv1d
python setup.py install

cd ..

cd mamba-2.2.2
python setup.py install
```


### 🔍 Download initial I3D 
We use the Kinetics pretrained I3D model from the reposity [kinetics_i3d_pytorch](https://github.com/hassony2/kinetics_i3d_pytorch/blob/master/model/model_rgb.pth)

### 💡 Dataset Preparation

#### FineDiving
- Please download the dataset from the repository [FineDiving](https://github.com/xujinglin/FineDiving).
- The data structure should be:
```
$DATASET_ROOT
├── FineDiving
│   ├── FINADiving_MTL_256s
│   │   ├── FINADivingWorldCup2021_Men3m_final_r1
│   │   │   ├── 0
│   │   │   │   ├── 00489.jpg
│   │   │   │   ...
│   │   │   │   └── 00592.jpg
│   │   │   ...
│   │   │   └── 11
│   │   │       ├── 14425.jpg
│   │   │       ...
│   │   │       └── 14542.jpg
│   │   ├── ...
│   │   └── FullMenSynchronised10mPlatform_Tokyo2020Replays_2
│   │       ├── 0
│   │       ...
│   │       └── 16
│   │
│   └── Annotations
│       ├── FineDiving_coarse_annotation.pkl
│       ├── FineDiving_fine-grained_annotation.pkl
│       ├── fine-grained_annotation_aqa.pkl
│       ├── Sub_action_Types_Table.pkl
│       ├── test_split.pkl
│       └── train_split.pkl

```


#### LOGO
- Please download the dataset from the repository [LOGO](https://github.com/shiyi-zh0408/LOGO).
- After downloading, run the data preparation script to process the dataset:

```bash
python data_prepare_LOGO.py
```
The data structure should be:
```
$DATASET_ROOT
├── LOGO
│   ├── Video_result                     
│   │   ├── WorldChampionship2019_free_final
│   │   │   ├── 0
│   │   │   │   ├── 00000.jpg
│   │   │   │   ...
│   │   │   │   └── 06249.jpg
│   │   │   ├── ...
│   │   │   └── 11
│   │   │       ├── 00000.jpg
│   │   │       ...
│   │   │       └── 06249.jpg
│   │   ├── ...
│   │   └── WorldChampionship2022_free_final
│   │       ├── 0
│   │       ├── ...
│   │       └── 7
│   ├── LOGO_Anno&Split                  
│   │   ├── anno_dict.pkl
│   │   ├── formation_dict.pkl
│   │   ├── test_split3.pkl
│   │   └── train_split3.pkl
│   └── i3d_features_LOGO.pkl            
```

## ✅ Train
To train the model, please run:
```bash
python main.py
```

## ✅ Test
To test the trained model, please set `test: True` in config and run:
```bash
python main.py
```


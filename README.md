# RVINN
**RNA Velocity-Informed Neural Networks (RVINN)** is a PINN-based framework for modeling mRNA dynamics and inferring transcriptional and post-transcriptional regulation.

---
<div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start;">

  <figure style="text-align: center;">
    <figcaption style="top: 5px;">RVINN Overview</figcaption>
    <img src="https://github.com/omuto/RVINN/blob/main/readme_fig/model_overview_github.png?raw=true" alt="Model" style="width: 150px;">
  </figure>

  <figure style="text-align: center;">
    <figcaption style="top: 5px;">Transcriptional Ripple</figcaption>
    <img src="https://github.com/omuto/RVINN/blob/main/readme_fig/Transcriptional_Ripple_animation.gif?raw=true" alt="Transcriptional Ripple" style="width: 200px;">
  </figure>

</div>


## Requirements
RVINN requires the following dependencies:
- **Python** 3.8 or later
- **PyTorch** 2.0.0 or later

Note: RVINN is primarily designed to run on **CPU** for both training and inference.

If PyTorch is not installed, please refer to the official installation guide:
[PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

## Installation (RVINN)

```console
pip install git+https://github.com/omuto/RVINN
```

[Quick demo](https://github.com/omuto/RVINN/blob/main/demo/rvinn_demo.ipynb)


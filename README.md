# Efficient Anomaly Detection using Diffusion Models
Repo for final project for MIT 6.5940 TinyML and Efficient Deep Learning Computing

This project is based on [Anomalies-by-Synthesis: Anomaly Detection using Generative Diffusion Models for Off-Road Navigation](https://bit.ly/anomalies-by-synthesis). This project proposes a multi-faceted optimization strategy tailored to the computational bottlenecks of the diffusion pipeline for anomaly detection.

The repo is built based on [anomalies-by-synthesis](https://github.com/siddancha/anomalies-by-synthesis).

## important files
Test environment: Run `experiment/pipe_way.py`.

Training
- training script: `diffunc/train/rugd_full_id_train.py`

Inference
- inference script: `experiment/pipe_way.py`
- inference functions: `experiment/pipe.py`
   - `get_generated_img` function: perform guided diffusion to generate image
   - it calls `diffusion.p_sample_loop` where  `diffusion` is a class of  `GaussianDiffusion`

Evaluation
- if it's your first time running the evaluation, run `experiment/freq_label.py` first
- evaluation script: `experiment/eval_way.py`
- evaluation functions: `experiment/eval.py`
- diffusion inference (guided diffusion) functions: `diffunc/guided_diffusion.py`

## Install instructions

1. Install [Git LFS](https://git-lfs.com/).

2. Clone repository and `cd` into it.
   ```bash
   git clone --recursive https://github.com/siddancha/diffunc.git
   cd /path/to/diffunc
   ```

3. Create a new Python virtual environment
   ```bash
   python3 -m venv .venv --prompt=diffunc
   ```

4. Activate the virtual environment
   ```bash
   source .venv/bin/activate
   ```

5. Install [pdm](https://pdm.fming.dev/)
   ```bash
   pip install pdm
   ```

6. Install Python dependencies via pdm.
   ```bash
   pdm install --no-isolation
   ```

7. Create the following symlink for `featup`:
   ```bash
   ln -s `pwd`/.venv/lib/python3.10/site-packages/clip/bpe_simple_vocab_16e6.txt.gz .venv/lib/python3.10/site-packages/featup/featurizers/maskclip/
   ```

## Setting up the RUGD dataset
1. Create a folder (or symlinked folder) called `data` inside the `diffunc` repo.
   ```bash
   mkdir data
   ```

2. Download the [RUGD dataset](http://rugd.vision/).

3. Unzip the downloaded files and structure the dataset as follows:
   ```
   data/RUGD
   ├── RUGD_frames-with-annotations
         ├── creek, park-1, etc.          -- folders for each scene containing ".png" files from the RGB camera.
   ├── RUGD_annotations
         ├── creek, park-1, etc.          -- folders for each scene containing ".png" label color images, colored using the class palette.
         ├── RUGD_annotation-colormap.txt -- mapping containing a list of class id, class name and class color.
   ```

4. Run the dataset conversion scripts.
   ```bash
   ./scripts/make_ddpm_train_set_rugd_full.py
   ./scripts/make_ddpm_train_set_rugd_trail_trail_15.py
   ```


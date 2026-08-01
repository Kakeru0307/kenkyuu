# ViTex

**ViTex: Visual Texture Control for Multi-track Symbolic Music Generation via Discrete Diffusion Models**

This repository contains the official implementation of **ViTex**, a discrete diffusion-based model for controllable multi-track symbolic music generation with visual texture conditioning.

---

## 🧩 Environment Setup

We recommend using **Python 3.12**.

Install the dependencies with:

```bash
pip install torch tensorboard librosa muspy accelerate pydub
```

---

## 📦 Dataset and Checkpoints

Our processed dataset and pretrained checkpoint are available here:
👉 [Google Drive Folder](https://drive.google.com/drive/folders/1D9V-1nU13asJJ7W7FvG4jaWT5CABo_Dn?usp=sharing)

Contents:

* `d3pm.ckpt`: pretrained discrete diffusion model weights
* `all_pkl.tar.gz`: training set
* `pkl_test.tar.gz`: test set

---

## 🚀 Inference

1. Download and extract the checkpoint and test set.

2. Modify the file paths in `utils/inference_utils.py`, specifically in the functions:

   * `get_model()`
   * `get_dataset()`
     Update these paths to match where you extracted the files.

3. Run the provided inference script:

   ```bash
   bash run.sh
   ```

   This script randomly samples chord progressions and ViTex conditions from the test set, and calls `pipeline.py` to generate new music examples.

You can modify `pipeline.py` or the command-line arguments to adjust configurations such as:

* Conditional vs. unconditional generation
* Guidance scales for each condition
* Inpainting or continuation modes

---

## 🧠 Training

### Preparing the Training Dataset

If you wish to use our preprocessed dataset, simply extract the provided archive.
To train on your own MIDI collection, follow these steps inside the `data_preprocess` folder:

1. **Filter invalid MIDI files**

   ```bash
   python filter.py
   ```

   This filters out non-4/4 tracks, extreme BPM values, missing drum tracks, etc.
   (See the `process_midi()` function for detailed filtering rules.)

2. **Normalize tempo and instrumentation**

   ```bash
   python normalize.py
   ```

   This step standardizes the BPM to 120 and maps instruments to a predefined set of 12 categories.

3. **Preprocess multi-track data**

   ```bash
   python preprocess_multi.py
   ```

   Extracts chord and instrumentation information and saves them as `.pkl` files.

4. **Split into training and test sets**

   ```bash
   python split.py
   ```

   Splits the preprocessed data at the song level.

After these steps, you’ll have a directory containing `.pkl` files ready for training.

---

### Running Training

Set the path to your dataset in the `training_config` section of `train.py`, then launch training:

```bash
accelerate launch train.py
```

If you are using multiple GPUs, configure the `accelerate config` settings according to your hardware setup before launching.

---


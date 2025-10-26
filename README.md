# Ad Intelligence

## Setup

1. Run the setup script:
```bash
./setup.sh
```

2. Activate the virtual environment:
```bash
source .venv/bin/activate
```

The setup script will automatically download the ad assets and install all required dependencies.

3. Finally, install Tesseract:
```
brew install tesseract
```

## TPP-Gaze Submodule Setup

Required for the attention map features.

- Add submodule to Python path when running:
  ```bash
  export PYTHONPATH="$(pwd)/tppgaze:$PYTHONPATH"
  ```
- Download model files:
  ```
  bash tppgaze/download_models.sh
  ```
- Optional: set explicit paths (if stored elsewhere):
  ```bash
  export TPPGAZE_CFG="/abs/path/to/config.yaml"
  export TPPGAZE_CKPT="/abs/path/to/model_transformer.pth"
  ```
- Sanity checks:
  ```bash
  PYTHONPATH="$(pwd)/tppgaze:$PYTHONPATH" venv/bin/python test_attention_minimal.py
  PYTHONPATH="$(pwd)/tppgaze:$PYTHONPATH" venv/bin/python -m image.extraction
  ```

## Citation

This project uses the **TPP-Gaze** model for gaze dynamics and scanpath prediction.  
If you use or build upon this component, please cite the following work:

```bibtex
@inproceedings{damelio2025tppgaze,
  title     = {TPP-Gaze: Modelling Gaze Dynamics in Space and Time with Neural Temporal Point Processes},
  author    = {D'Amelio, Alessandro and Cartella, Giuseppe and Cuculo, Vittorio and Lucchi, Manuele and Cornia, Marcella and Cucchiara, Rita and Boccignone, Giuseppe},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year      = {2025}
}
````

For more information, visit the official repository:
[https://github.com/phuselab/tppgaze](https://github.com/phuselab/tppgaze)


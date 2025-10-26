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

1. Install brew dependencies:
```
brew install wget
```

2. Follow the setup instructions at https://github.com/phuselab/tppgaze:
```
git clone https://github.com/phuselab/tppgaze
cd tppgaze
pip install -r requirements
```
(If you are having issues with tppgaze, remove tppgaze, then redo this instruction (2).)

3. Within the tppgaze folder still, run the following.
```
bash download_models.sh
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


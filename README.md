# Gaze Following - Determinar a direção do olhar
### => based on head pose estimation

Determinar a direção do olhar pode ter diversas aplicações em diferentes setores da economia. Este repositório tem como objetivo estimar a direção do olhar com base na estimativa de pose da cabeça.
O dataset repopulado está disponível no formato .txt na pasta /dataset.
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required packages.

```bash
pip install -r requirements.txt
```
Download model from [GoogleDrive](https://drive.google.com/file/d/1hGsl9vyA9Mg-r-9oKSJ2lqUgErc3uOLE/view?usp=sharing) to:
```bash
model\
```
## Usage
Change video path, or replace the string with number 0 as integer to work with webcam, and run:
```bash
python test_realtime.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
Licença para uso não-comercial, apenas fins academicos. Repositório baseado em [GazeFollowing](https://github.com/svip-lab/GazeFollowing)
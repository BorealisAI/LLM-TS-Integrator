## Project Description

We introduce LLM-TS Integrator, a framework that effectively integrates the capabilities of LLMs with traditional TS modeling.

## Usage

1. **Set Up Environment:**
   Ensure Python 3.8 is installed and then set up the required libraries using:
   ```
   pip install -r requirements.txt
   ```
2. **Data Preparation:**
   Download pre-processed datasets from [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy). Place the files in `./dataset`. Below is an overview of the supported datasets:
   <p align="center">
   <img src=".\pic\dataset.png" height = "200" alt="" align=center />
   </p>

3. **Model Training and Evaluation:**
   Execute scripts for various tasks using commands in `./scripts/`:
   - Long-term forecast: `bash ./scripts/long_term_forecast/EXP1.sh`
   - Short-term forecast: `bash ./scripts/short_term_forecast/EXP1.sh`
   - Imputation: `bash ./scripts/imputation/ETT_script/EXP1.sh`
   - Anomaly detection: `bash ./scripts/anomaly_detection/PSM/EXP1.sh`
   - Classification: `bash ./scripts/classification/EXP1.sh`

## Acknowledgements

Special thanks to the **TimesNet library** ([TSlib](https://github.com/thuml/Time-Series-Library/tree/main)) for their extensive resources that supported this project.
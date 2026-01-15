# BigEarthNet Distributed Deep Learning

A distributed deep learning project for large-scale remote sensing data analysis using the BigEarthNet v2.0 dataset.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-Package%20Manager-DE5FE9?style=flat&logo=astral&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![PyArrow](https://img.shields.io/badge/PyArrow-Data-00C7B7?style=flat&logo=apache&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Analysis-150458?style=flat&logo=pandas&logoColor=white)
![AWS S3](https://img.shields.io/badge/AWS%20S3-Storage-569A31?style=flat&logo=amazons3&logoColor=white)
![Boto3](https://img.shields.io/badge/Boto3-SDK-FF9900?style=flat&logo=amazonaws&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Petastorm](https://img.shields.io/badge/Petastorm-Data%20Pipeline-00ADD8?style=flat&logo=apache&logoColor=white)

s3+petastorm + tensorflow pipeline 

<img width="1291" height="371" alt="image" src="https://github.com/user-attachments/assets/6e2ebb0c-e5a4-4865-913b-c46d39f2cd7d" />


## About the Project

This project explores distributed computing techniques for deep learning on big data using the BigEarthNet v2.0 dataset. BigEarthNet is a large-scale benchmark archive for remote sensing image analysis, consisting of 549,488 pairs of Sentinel-1 and Sentinel-2 satellite image patches covering various European countries.

## About BigEarthNet Dataset

**BigEarthNet v2.0** is a benchmark dataset developed by TU Berlin, consisting of:

- Coverage of 10 European countries (Austria, Belgium, Finland, Ireland, Kosovo, Lithuania, Luxembourg, Portugal, Serbia, and Switzerland)
- Images acquired between June 2017 and May 2018
- Multi-label land-cover classification based on CORINE Land Cover 2018 (CLC2018)
- Pixel-level reference maps
- **Total Files**: 8,242,325 files | 144.1 GB | TIF format

### Dataset Components

- **BigEarthNet-S2**: Sentinel-2 multispectral image patches 
- **BigEarthNet-S1**: Sentinel-1 SAR image patches 
- **Reference Maps**: Pixel-level land-cover maps
- **Metadata**: metadata in Parquet format

For more information, visit: https://bigearth.net/

## Project Structure

```
bigearthnet-dl-distributed/
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── scripts/            # Python scripts (converted from notebooks or standalone)
├── experiments/        # Experiment configurations and results
├── results/            # Generated reports and analysis outputs
├── main.py             # entry point
└── pyproject.toml      # Project dependencies and configuration
```

## Setup

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer

### Install uv

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Using pip:**
```bash
pip install uv
```

### Install Project Dependencies

Once `uv` is installed, set up the project:

```bash
# Clone the repository (if not already)
git clone https://github.com/kshitijrajsharma/fractal-ml-distributed-computing.git
```

```bash
cd bigearthnet-dl-distributed
uv sync
```

This installs the project and makes the following commands available:
- `uv run gen-metadata` - Generate metadata with S3 paths
- `uv run check-s3` - Validate file availability
- `uv run to-petastorm` - Convert TIF to TFRecord
- `uv run train-model` - Train segmentation model

## Usage

See [scripts/Readme.md](scripts/Readme.md) for detailed usage examples and parameters.

Use JupyterLab, VS Code, or Jupyter Notebook to work with notebooks.
jupyter nbconvert --to script notebooks/01-preliminary.ipynb --output-dir=scripts/
```

**Run scripts:**
```bash
python scripts/01-preliminary.py
```

## Data Access

For this project, BigEarthNet data has been downloaded and stored on S3. You can download the dataset from [bigearth.net](https://bigearth.net/) and organize it in the following structure:

```
s3://your-bucket/bigearthnet/
├── metadata.parquet
├── BigEarthNet-S1/
├── BigEarthNet-S2/
└── Reference_Maps/
```
## Resources

- [BigEarthNet Official Website](https://bigearth.net/)
- [Dataset Documentation](https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf)
- [BigEarthNet Pipeline (GitHub)](https://github.com/rsim-tu-berlin/bigearthnet-pipeline)
- [uv Documentation](https://docs.astral.sh/uv/)


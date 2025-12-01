# Matrix Analysis Lab

A desktop application for image denoising using Random Matrix Theory (RMT) and eigenvalue analysis based on the Marchenko-Pastur (M-P) distribution.

## Overview

Matrix Analysis Lab is a powerful tool that applies random matrix theory principles to image processing. It uses eigenvalue decomposition and spectral filtering techniques to denoise images while preserving important features. The application provides a visual interface to compare different denoising methods and analyze eigenvalue distributions.

## Features

### Core Functionality
- **Multiple Denoising Methods**:
  - Marchenko-Pastur (M-P) Law Denoising
  - Generalized Covariance Denoising (with auto-optimization)
  - Manual parameter control for advanced users
  - M-P Standard Deviation-based methods

- **Noise Types Supported**:
  - Gaussian noise
  - Laplacian noise (heavy-tailed distribution)
  - Mixture of Gaussians
  - Custom noise parameters

- **Eigenvalue Analysis**:
  - Real-time eigenvalue distribution visualization
  - Interactive plots with histogram and KDE smoothing
  - Comprehensive statistical summaries (mean, median, std dev, percentiles)
  - Export eigenvalues to CSV format

- **Batch Processing**:
  - Process entire folders of images
  - Compare multiple denoising methods side-by-side
  - Navigate through processed results
  - Save output to organized folders

### Advanced Features
- **GPU Acceleration**: Optional CuPy support for faster matrix operations
- **Interactive UI**: Built with pywebview for native desktop performance
- **Visualization**: Plotly-powered interactive charts
- **File Management**: Built-in temp file manager to organize outputs

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional Dependencies

**For GPU Acceleration** (requires NVIDIA GPU with CUDA):
```bash
pip install cupy-cuda11x
```

**For Browser Automation Features**:
```bash
pip install playwright
playwright install
```

## Usage

### Running the Application

```bash
python app.py
```

Or use the compiled executable:
```bash
./dist/app.exe
```

### Building Executable

To create a standalone executable using PyInstaller:

```bash
pyinstaller app.spec --clean
```

The executable will be created in the `dist/` folder (~78 MB optimized build).

## How It Works

### Random Matrix Theory & Marchenko-Pastur Distribution

The application leverages random matrix theory, specifically the Marchenko-Pastur (M-P) distribution, to distinguish between signal and noise in image data.

**Key Concepts**:

1. **Eigenvalue Decomposition**: Images are treated as matrices and decomposed using Singular Value Decomposition (SVD)

2. **M-P Law**: For a random matrix with independent entries, the eigenvalue distribution follows the Marchenko-Pastur law. This theoretical distribution helps identify which eigenvalues correspond to noise

3. **Spectral Filtering**: By removing eigenvalues that fall within the M-P noise distribution, the algorithm reconstructs a denoised image containing only the signal components

4. **Generalized Methods**: Extended versions that account for non-Gaussian noise and heavy-tailed distributions using parameters α (aspect ratio) and β (distribution shape)

### Denoising Methods

- **Method 1: M-P Basic**: Standard Marchenko-Pastur denoising
- **Method 2: M-P Auto**: Auto-optimized parameters using SSIM
- **Method 3: Gen Cov Auto**: Generalized covariance with automatic parameter estimation
- **Method 4: Gen Cov Manual**: Full manual control over σ², α, and β parameters
- **Method 5: M-P Std Dev**: Uses standard deviation of top eigenvalues for threshold

## Application Structure

### Main Components

- **Bridge Class** (app.py:1143): Backend API that handles all image processing operations
- **Denoising Functions**: Various implementations of M-P and generalized methods
- **Eigenvalue Analysis**: Compute and visualize eigenvalue distributions
- **File Management**: Handle folder uploads, processing, and temp file cleanup

### UI Sections

1. **Folder Management**: Create folders, upload files, browse images
2. **Processing Panel**: Configure denoising methods and noise parameters
3. **Results Display**: 2x2 grid showing original, noisy, and two denoised versions
4. **Eigenvalue Viewer**: Click any image to analyze its eigenvalue distribution
5. **Temp Files Manager**: View and manage output folders

## Technical Details

### Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Matrix operations and numerical computing |
| scipy | Statistical functions, optimization, and filtering |
| Pillow | Image I/O and manipulation |
| sympy | Symbolic mathematics for cubic equation solving |
| pywebview | Cross-platform desktop GUI |
| scikit-learn | KDE (Kernel Density Estimation) for smoothing |
| psutil | System monitoring |

### Performance

- **Single-threaded** processing with numpy/scipy
- **GPU acceleration** available with CuPy (significant speedup for large images)
- **Memory efficient** with data centralization and normalization
- **Optimized builds** using PyInstaller with UPX compression

## File Organization

```
.
├── app.py                 # Main application
├── app.spec              # PyInstaller configuration
├── requirements.txt      # Python dependencies
├── icon.ico             # Application icon
├── .random_matrix/      # User data folder (auto-created)
│   └── [user folders]   # Uploaded image folders
└── dist/                # Built executables (after compilation)
    └── app.exe          # Standalone application
```

## Configuration

### PyInstaller Optimization

The `app.spec` file is optimized to exclude unnecessary libraries:
- Excludes ML frameworks (torch, tensorflow, transformers)
- Excludes unused GUI frameworks (tkinter, Qt, matplotlib)
- Excludes web frameworks (gradio, fastapi)
- Keeps essential scientific libraries (numpy, scipy, sklearn)
- Enables UPX compression for smaller executable size

### Build Size Optimization

- **Original build**: ~458 MB
- **Optimized build**: ~78 MB (83% reduction)
- Key exclusions documented in app.spec:13-76

## Troubleshooting

### Common Issues

**Issue**: Module not found errors
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: GPU acceleration not working
- **Solution**: Install CuPy with correct CUDA version: `pip install cupy-cuda11x`

**Issue**: Executable fails to run
- **Solution**: Rebuild using `pyinstaller app.spec --clean`

**Issue**: scipy.sparse or pydoc errors
- **Solution**: Do not exclude scipy submodules in app.spec (they have complex interdependencies)

## Theory & References

This application implements concepts from:
- Marchenko-Pastur law for eigenvalue distribution of random matrices
- Spectral filtering techniques for signal/noise separation
- Generalized random matrix theory for heavy-tailed distributions
- Singular Value Decomposition (SVD) for image denoising

**Related Research**:
- V. A. Marčenko and L. A. Pastur, "Distribution of eigenvalues for some sets of random matrices," Math. USSR-Sbornik, 1967
- Image denoising via spectral filtering and random matrix theory
- Applications of RMT in signal processing and data science

## License

This project is provided as-is for educational and research purposes.

## Version

- **Application**: Matrix Analysis Lab
- **Python**: 3.12.3
- **PyInstaller**: 6.11.0
- **Build Date**: 2025-12-01

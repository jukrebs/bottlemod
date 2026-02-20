#!/bin/bash
# BottleMod Custom SciPy Setup Script
#
# This script creates a virtual environment with a custom patched SciPy v1.15.1
# Supports Ubuntu/Debian (apt-get) and Arch Linux (pacman)

set -euo pipefail  # Exit on any error, undefined vars, pipe failures

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME=".venv"
ENV_PATH="$PROJECT_ROOT/$ENV_NAME"

echo "========================================"
echo "BottleMod Custom SciPy Setup"
echo "========================================"

# Detect OS and package manager
echo "Detecting system package manager..."
if command -v apt-get &> /dev/null; then
    PACKAGE_MANAGER="apt-get"
    echo "Detected: Ubuntu/Debian (apt-get)"
elif command -v pacman &> /dev/null; then
    PACKAGE_MANAGER="pacman"
    echo "Detected: Arch Linux (pacman)"
else
    echo "ERROR: Neither apt-get nor pacman found. Unsupported distribution."
    exit 1
fi

# Install system requirements based on package manager
echo "Installing system requirements via $PACKAGE_MANAGER..."
if [ "$PACKAGE_MANAGER" = "apt-get" ]; then
    sudo DEBIAN_FRONTEND=noninteractive apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y git gcc gfortran libblas-dev libopenblas-dev liblapack-dev libffi-dev pkg-config build-essential python3-dev python3-venv ninja-build
elif [ "$PACKAGE_MANAGER" = "pacman" ]; then
    sudo pacman -S --needed -y git gcc gcc-fortran openblas pkgconf base-devel
fi
echo "System requirements OK!"

# Create virtual environment
echo "Creating virtual environment: $ENV_NAME"
if [ -d "$ENV_PATH" ]; then
    echo "Removing existing environment..."
    rm -rf "$ENV_PATH"
fi

python3 -m venv "$ENV_PATH"
source "$ENV_PATH/bin/activate"

echo "Virtual environment created and activated!"

# Upgrade pip and install build dependencies
echo "Installing build dependencies..."
pip install --upgrade pip wheel setuptools

# Install NumPy first (required for SciPy build)
# Using the exact numpy requirement from scipy pyproject.toml
pip install "numpy>=2.0.0,<2.5"

# Install SciPy build dependencies from pyproject.toml (v1.15.1)
echo "Installing Meson build system dependencies..."
pip install \
    "meson-python>=0.15.0,<0.20.0" \
    "Cython>=3.0.8,<3.1.0" \
    "pybind11>=2.13.2,<2.14.0" \
    "pythran>=0.14.0,<0.18.0" \
    setuptools_scm

# Create temporary directory for SciPy build
TEMP_DIR=$(mktemp -d)
SCIPY_DIR="$TEMP_DIR/scipy"

echo "Cloning SciPy v1.15.1..."
git clone --branch v1.15.1 --depth 1 https://github.com/scipy/scipy.git "$SCIPY_DIR"

cd "$SCIPY_DIR"
echo "Initializing submodules..."
git submodule update --init

# Apply custom _ppoly.pyx patch
echo "Applying custom _ppoly.pyx patch..."
CUSTOM_PPOLY_URL="https://raw.githubusercontent.com/bottlemod/bottlemod/refs/heads/main/custom_scipy/_ppoly_v1.15.1.pyx"
TARGET_PPOLY="$SCIPY_DIR/scipy/interpolate/_ppoly.pyx"

if command -v curl &> /dev/null; then
    curl -fsSL "$CUSTOM_PPOLY_URL" -o "$TARGET_PPOLY"
elif command -v wget &> /dev/null; then
    wget -qO "$TARGET_PPOLY" "$CUSTOM_PPOLY_URL"
else
    echo "ERROR: Neither curl nor wget found. Cannot download custom _ppoly.pyx patch."
    exit 1
fi

echo "Custom _ppoly.pyx patch applied!"

# Build and install SciPy
echo "Building custom SciPy..."
# Install SciPy using pip with --no-build-isolation (required for Meson builds)
pip install . --no-build-isolation

# Return to project directory
cd "$PROJECT_ROOT"

echo "Installing project dependencies..."
pip install -r requirements.txt

# Cleanup temporary directory
rm -rf "$TEMP_DIR"

echo "Setup complete! Activate the environment with: source $ENV_PATH/bin/activate"

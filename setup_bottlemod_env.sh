#!/bin/bash
# BottleMod Custom SciPy Setup Script
#
# This script creates a virtual environment with a custom patched SciPy v1.15.1

set -e  # Exit on any error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME=".venv"
ENV_PATH="$PROJECT_ROOT/$ENV_NAME"

echo "========================================"
echo "BottleMod Custom SciPy Setup"
echo "========================================"

# Check system requirements
# Arch linux
echo "Installing system requirements... on Arch Linux"
sudo pacman -S --needed git gcc gcc-fortran openblas pkgconf base-devel
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
CUSTOM_PPOLY="$PROJECT_ROOT/bottlemod/custom_scipy/_ppoly_v1.15.1.pyx"
TARGET_PPOLY="$SCIPY_DIR/scipy/interpolate/_ppoly.pyx"

if [ ! -f "$CUSTOM_PPOLY" ]; then
    echo "ERROR: Custom _ppoly.pyx file not found: $CUSTOM_PPOLY"
    echo "Make sure the file exists in bottlemod/custom_scipy/_ppoly_v1.15.1.pyx"
    exit 1
fi

# Backup original and apply patch
# cp "$TARGET_PPOLY" "$TARGET_PPOLY.original"
cp "$CUSTOM_PPOLY" "$TARGET_PPOLY"
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

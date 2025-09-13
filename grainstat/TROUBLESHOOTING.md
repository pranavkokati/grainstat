# GrainStat Installation Troubleshooting Guide

If you're encountering the error `ModuleNotFoundError: No module named 'grainstat.core'`, follow these steps to resolve it:

## Quick Fix

1. **Uninstall and reinstall in editable mode:**
   ```bash
   pip uninstall grainstat
   pip install -e .
   ```

2. **Or install from the current directory:**
   ```bash
   pip install .
   ```

## Step-by-Step Troubleshooting

### 1. Verify Python Version
```bash
python --version
```
Make sure you're using Python 3.8 or later.

### 2. Check Current Installation
```bash
pip show grainstat
```

### 3. Uninstall Previous Installation
```bash
pip uninstall grainstat
```

### 4. Install Dependencies First
```bash
pip install numpy scipy scikit-image matplotlib pandas Pillow seaborn
```

### 5. Install GrainStat
```bash
# From the grainstat directory
pip install -e .

# Or for regular installation
pip install .
```

### 6. Verify Installation
```bash
python test_installation.py
```

## Alternative Installation Methods

### Method 1: Using setup.py directly
```bash
python setup.py develop
```

### Method 2: Using pip with specific options
```bash
pip install --force-reinstall --no-deps .
pip install -r requirements.txt
```

### Method 3: Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv grainstat_env

# Activate it
# On macOS/Linux:
source grainstat_env/bin/activate
# On Windows:
grainstat_env\Scripts\activate

# Install grainstat
pip install -e .
```

## Common Issues and Solutions

### Issue 1: Package not found
**Error:** `No module named 'grainstat'`
**Solution:** Make sure you're in the correct directory and use `pip install -e .`

### Issue 2: Submodules not found
**Error:** `No module named 'grainstat.core'`
**Solution:** This indicates the package structure wasn't properly installed. Use the steps above.

### Issue 3: Permission errors
**Error:** `Permission denied`
**Solution:** Use `pip install --user .` or install in a virtual environment.

### Issue 4: Conflicting installations
**Error:** Mixed import errors
**Solution:** 
```bash
pip uninstall grainstat
pip cache purge
pip install -e .
```

## Verification Test

After installation, run this test:

```python
# Test basic import
import grainstat
print("✓ grainstat imported")

# Test core modules
from grainstat.core import ImageLoader
print("✓ grainstat.core imported")

# Test main class
from grainstat import GrainAnalyzer
analyzer = GrainAnalyzer()
print("✓ GrainAnalyzer created successfully")
```

## Still Having Issues?

1. Check your Python environment:
   ```bash
   which python
   which pip
   ```

2. Make sure you're in the right directory (where setup.py is located)

3. Try installing in a fresh virtual environment

4. Check for any missing dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. If all else fails, try:
   ```bash
   python -c "import sys; print(sys.path)"
   ```
   And make sure your grainstat installation directory is in the path.

## Contact

If you continue to have issues, please provide:
- Your Python version
- Your operating system
- The exact error message
- Output of `pip list | grep grainstat`

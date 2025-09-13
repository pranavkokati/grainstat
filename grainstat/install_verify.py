#!/usr/bin/env python3
"""
Installation verification and troubleshooting script for GrainStat
"""

import sys
import subprocess
import pkg_resources

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8 or later.")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor} is compatible")
        return True

def check_package_installed():
    """Check if grainstat is installed"""
    try:
        pkg_resources.get_distribution('grainstat')
        print("✅ grainstat package is installed")
        return True
    except pkg_resources.DistributionNotFound:
        print("❌ grainstat package is not installed")
        return False

def install_grainstat():
    """Install grainstat package"""
    print("Installing grainstat...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        print("✅ grainstat installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False

def reinstall_grainstat():
    """Reinstall grainstat package"""
    print("Reinstalling grainstat...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "grainstat"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        print("✅ grainstat reinstalled successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Reinstallation failed: {e}")
        return False

def main():
    print("GrainStat Installation Troubleshoot")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check if package is installed
    if not check_package_installed():
        print("\nAttempting to install grainstat...")
        if not install_grainstat():
            return
    
    # Test imports
    print("\nTesting imports...")
    try:
        exec(open('test_installation.py').read())
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        print("\nAttempting to reinstall...")
        if reinstall_grainstat():
            print("Please run this script again to verify the installation.")
        return
    
    print("\n🎉 GrainStat is correctly installed and working!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script to verify grainstat installation
"""

def test_imports():
    """Test that all major components can be imported"""
    print("Testing grainstat imports...")
    
    try:
        import grainstat
        print("✓ grainstat package imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import grainstat: {e}")
        return False
    
    try:
        from grainstat import GrainAnalyzer
        print("✓ GrainAnalyzer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import GrainAnalyzer: {e}")
        return False
    
    try:
        from grainstat.core import ImageLoader
        print("✓ grainstat.core modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import grainstat.core: {e}")
        return False
    
    try:
        from grainstat.plugins import feature
        print("✓ grainstat.plugins imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import grainstat.plugins: {e}")
        return False
    
    try:
        from grainstat import core, export, plugins, processing, visualization
        print("✓ All submodules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import submodules: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from grainstat import GrainAnalyzer
        analyzer = GrainAnalyzer()
        print("✓ GrainAnalyzer instance created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to create GrainAnalyzer: {e}")
        return False

if __name__ == "__main__":
    print("GrainStat Installation Test")
    print("=" * 30)
    
    import_success = test_imports()
    functionality_success = test_basic_functionality()
    
    if import_success and functionality_success:
        print("\n🎉 All tests passed! GrainStat is correctly installed.")
    else:
        print("\n❌ Some tests failed. Please check the installation.")
        exit(1)

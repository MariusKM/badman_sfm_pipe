import sys
try:
    import pycolmap
    print(f"✓ pycolmap imported successfully")
    print(f"  Version: {pycolmap.__version__}")
    
    # Test basic classes
    reconstruction = pycolmap.Reconstruction()
    print(f"✓ Reconstruction class works")
    
    camera = pycolmap.Camera()
    print(f"✓ Camera class works")
    
    print("\n✓ All pycolmap tests passed!")
    
except ImportError as e:
    print(f"✗ Failed to import pycolmap: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error testing pycolmap: {e}")
    sys.exit(1)
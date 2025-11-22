"""
ComfyUI-UniRig Prestartup Script

Handles setup tasks before node loading:
- Copy asset files to input/3d/
- Copy workflow examples to user/default/workflows/ with "UniRig -" prefix
- Create necessary directories
"""

import os
import shutil
import json
from pathlib import Path

# Get paths relative to this script
SCRIPT_DIR = Path(__file__).parent.absolute()
COMFYUI_DIR = SCRIPT_DIR.parent.parent  # custom_nodes/ComfyUI-UniRig -> custom_nodes -> ComfyUI

# Source directories
ASSETS_DIR = SCRIPT_DIR / "assets"
WORKFLOWS_DIR = SCRIPT_DIR / "workflows"

# Target directories (using relative paths from ComfyUI root)
INPUT_3D_DIR = COMFYUI_DIR / "input" / "3d"
USER_WORKFLOWS_DIR = COMFYUI_DIR / "user" / "default" / "workflows"


def copy_asset_files():
    """Copy all asset files to input/3d/ directory"""
    try:
        # Create target directory
        INPUT_3D_DIR.mkdir(parents=True, exist_ok=True)

        if not ASSETS_DIR.exists():
            print(f"[UniRig] Warning: Assets directory not found at {ASSETS_DIR}")
            return

        # Copy all files from assets directory
        for asset_file in ASSETS_DIR.iterdir():
            if asset_file.is_file():
                target_file = INPUT_3D_DIR / asset_file.name
                
                if not target_file.exists():
                    shutil.copy2(str(asset_file), str(target_file))
                    print(f"[UniRig] Copied {asset_file.name} to {target_file}")
                else:
                    print(f"[UniRig] {asset_file.name} already exists at {target_file}")

    except Exception as e:
        print(f"[UniRig] Error copying asset files: {e}")
        import traceback
        traceback.print_exc()


# Execute setup tasks
try:
    print("[UniRig] Running prestartup script...")
    copy_asset_files()
    print("[UniRig] Prestartup script completed.")
except Exception as e:
    print(f"[UniRig] Error during prestartup: {e}")
    import traceback
    traceback.print_exc()

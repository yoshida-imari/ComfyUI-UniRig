"""
UniRig nodes for ComfyUI

Provides state-of-the-art skeleton extraction and rigging using the UniRig framework.
Self-contained with bundled Blender and UniRig code.
"""

import os
import sys
import subprocess
import tempfile
import numpy as np
import trimesh
from pathlib import Path
import folder_paths


# Get paths relative to this file
NODE_DIR = Path(__file__).parent.absolute()
LIB_DIR = NODE_DIR / "lib"
UNIRIG_PATH = str(LIB_DIR / "unirig")
BLENDER_SCRIPT = str(LIB_DIR / "blender_extract.py")
BLENDER_PARSE_SKELETON = str(LIB_DIR / "blender_parse_skeleton.py")

# Find Blender executable
BLENDER_DIR = LIB_DIR / "blender"
BLENDER_EXE = None
if BLENDER_DIR.exists():
    blender_bins = list(BLENDER_DIR.rglob("blender"))
    if blender_bins:
        BLENDER_EXE = str(blender_bins[0])
        print(f"[UniRig] Found Blender: {BLENDER_EXE}")

# Install Blender if not found
if not BLENDER_EXE:
    print("[UniRig] Blender not found, installing...")
    try:
        from .install_blender import install_blender
        BLENDER_EXE = install_blender(target_dir=BLENDER_DIR)
        if BLENDER_EXE:
            print(f"[UniRig] Blender installed: {BLENDER_EXE}")
        else:
            print("[UniRig] Warning: Blender installation failed")
    except Exception as e:
        print(f"[UniRig] Warning: Could not install Blender: {e}")

# Add local UniRig to path
if UNIRIG_PATH not in sys.path:
    sys.path.insert(0, UNIRIG_PATH)


def normalize_skeleton(vertices: np.ndarray) -> np.ndarray:
    """Normalize skeleton vertices to [-1, 1] range."""
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    center = (min_coords + max_coords) / 2
    vertices = vertices - center
    scale = (max_coords - min_coords).max() / 2
    if scale > 0:
        vertices = vertices / scale
    return vertices


class UniRigExtractSkeleton:
    """
    Extract skeleton from mesh using UniRig (SIGGRAPH 2025).

    Uses ML-based approach for high-quality semantic skeleton extraction.
    Works on any mesh type: humans, animals, objects, cameras, etc.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 4294967295,  # numpy's max seed (2^32-1)
                               "tooltip": "Random seed for skeleton generation variation"}),
            },
            "optional": {
                "checkpoint": ("STRING", {
                    "default": "VAST-AI/UniRig",
                    "tooltip": "HuggingFace model ID or local path"
                }),
            }
        }

    RETURN_TYPES = ("SKELETON",)
    RETURN_NAMES = ("skeleton",)
    FUNCTION = "extract"
    CATEGORY = "UniRig"

    def extract(self, trimesh, seed, checkpoint="VAST-AI/UniRig"):
        """Extract skeleton using UniRig."""
        print(f"[UniRigExtractSkeleton] Starting skeleton extraction...")

        # Check if Blender is available
        if not BLENDER_EXE or not os.path.exists(BLENDER_EXE):
            raise RuntimeError(
                f"Blender not found. Please run install_blender.py or install manually."
            )

        # Check if UniRig is available
        if not os.path.exists(UNIRIG_PATH):
            raise RuntimeError(
                f"UniRig code not found at {UNIRIG_PATH}. "
                "The lib/unirig directory should contain the UniRig source code."
            )

        # Create temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.glb")
            # UniRig expects NPZ at: {npz_dir}/{basename}/raw_data.npz
            # Since input is "input.glb", basename is "input", so we need npz_dir to be tmpdir
            # and the NPZ will be at {tmpdir}/input/raw_data.npz
            npz_dir = os.path.join(tmpdir, "input")
            npz_path = os.path.join(npz_dir, "raw_data.npz")
            output_path = os.path.join(tmpdir, "skeleton.fbx")

            os.makedirs(npz_dir, exist_ok=True)

            # Export mesh to GLB
            print(f"[UniRigExtractSkeleton] Exporting mesh to {input_path}")
            print(f"[UniRigExtractSkeleton] Mesh has {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
            trimesh.export(input_path)
            print(f"[UniRigExtractSkeleton] ✓ Mesh exported")

            # Step 1: Extract/preprocess mesh with Blender
            print(f"[UniRigExtractSkeleton] Step 1: Preprocessing mesh with Blender...")
            blender_cmd = [
                BLENDER_EXE,
                "--background",
                "--python", BLENDER_SCRIPT,
                "--",
                input_path,
                npz_path,
                "50000"  # target face count
            ]

            try:
                result = subprocess.run(
                    blender_cmd,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.stdout:
                    print(f"[UniRigExtractSkeleton] Blender output:\n{result.stdout}")
                if result.stderr:
                    # Blender always outputs some stuff to stderr, filter out noise
                    stderr_lines = result.stderr.split('\n')
                    important_lines = [l for l in stderr_lines if 'error' in l.lower() or 'fail' in l.lower()]
                    if important_lines:
                        print(f"[UniRigExtractSkeleton] Blender warnings:\n" + '\n'.join(important_lines))

                # Check if NPZ was created (ignore return code, Blender might segfault after saving)
                if not os.path.exists(npz_path):
                    raise RuntimeError(f"Blender extraction failed: {npz_path} not created")

                print(f"[UniRigExtractSkeleton] ✓ Mesh preprocessed: {npz_path}")

            except subprocess.TimeoutExpired:
                raise RuntimeError("Blender extraction timed out (>2 minutes)")
            except Exception as e:
                print(f"[UniRigExtractSkeleton] Blender error: {e}")
                raise

            # Step 2: Run skeleton inference
            print(f"[UniRigExtractSkeleton] Step 2: Running skeleton inference...")
            run_cmd = [
                sys.executable, os.path.join(UNIRIG_PATH, "run.py"),
                "--task", os.path.join(UNIRIG_PATH, "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"),
                "--seed", str(seed),
                "--input", input_path,
                "--output", output_path,
                "--npz_dir", tmpdir,  # This should match where NPZ is: tmpdir/input/raw_data.npz
            ]

            print(f"[UniRigExtractSkeleton] Running: {' '.join(run_cmd)}")
            print(f"[UniRigExtractSkeleton] Using Blender: {BLENDER_EXE}")

            # Set up environment with Blender path for internal FBX export
            env = os.environ.copy()
            env['BLENDER_EXE'] = BLENDER_EXE
            print(f"[UniRigExtractSkeleton] Set BLENDER_EXE environment variable for FBX export")

            try:
                result = subprocess.run(
                    run_cmd,
                    cwd=UNIRIG_PATH,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=180
                )
                if result.stdout:
                    print(f"[UniRigExtractSkeleton] Inference stdout:\n{result.stdout}")
                if result.stderr:
                    print(f"[UniRigExtractSkeleton] Inference stderr:\n{result.stderr}")

                if result.returncode != 0:
                    print(f"[UniRigExtractSkeleton] ✗ Inference failed with exit code {result.returncode}")
                    raise RuntimeError(f"Inference failed with exit code {result.returncode}")

                print(f"[UniRigExtractSkeleton] ✓ Inference completed successfully")

            except subprocess.TimeoutExpired:
                raise RuntimeError("Inference timed out (>3 minutes)")
            except Exception as e:
                print(f"[UniRigExtractSkeleton] Inference error: {e}")
                raise

            # Load and parse FBX output
            if not os.path.exists(output_path):
                tmpdir_contents = os.listdir(tmpdir)
                print(f"[UniRigExtractSkeleton] ✗ Output FBX not found: {output_path}")
                print(f"[UniRigExtractSkeleton] Temp directory contents: {tmpdir_contents}")
                raise RuntimeError(
                    f"UniRig did not generate output file: {output_path}\n"
                    f"Temp directory contents: {tmpdir_contents}\n"
                    f"Check stdout/stderr above for details"
                )

            print(f"[UniRigExtractSkeleton] ✓ Found output FBX: {output_path}")
            fbx_size = os.path.getsize(output_path)
            print(f"[UniRigExtractSkeleton] FBX file size: {fbx_size} bytes")

            print(f"[UniRigExtractSkeleton] Step 3: Parsing FBX output with Blender...")
            skeleton_npz = os.path.join(tmpdir, "skeleton_data.npz")

            # Use Blender to parse skeleton from FBX
            parse_cmd = [
                BLENDER_EXE,
                "--background",
                "--python", BLENDER_PARSE_SKELETON,
                "--",
                output_path,
                skeleton_npz,
            ]

            try:
                result = subprocess.run(
                    parse_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.stdout:
                    print(f"[UniRigExtractSkeleton] Blender parse output:\n{result.stdout}")
                if result.stderr:
                    stderr_lines = result.stderr.split('\n')
                    important_lines = [l for l in stderr_lines if 'error' in l.lower() or 'fail' in l.lower()]
                    if important_lines:
                        print(f"[UniRigExtractSkeleton] Blender parse warnings:\n" + '\n'.join(important_lines))

                if not os.path.exists(skeleton_npz):
                    print(f"[UniRigExtractSkeleton] ✗ Skeleton NPZ not found: {skeleton_npz}")
                    raise RuntimeError(f"Skeleton parsing failed: {skeleton_npz} not created")

                print(f"[UniRigExtractSkeleton] ✓ Skeleton NPZ created")

            except subprocess.TimeoutExpired:
                raise RuntimeError("Skeleton parsing timed out (>1 minute)")
            except Exception as e:
                print(f"[UniRigExtractSkeleton] Skeleton parse error: {e}")
                raise

            # Load skeleton data
            print(f"[UniRigExtractSkeleton] Loading skeleton data from NPZ...")
            skeleton_data = np.load(skeleton_npz, allow_pickle=True)
            print(f"[UniRigExtractSkeleton] NPZ contains keys: {list(skeleton_data.keys())}")
            vertices = skeleton_data['vertices']
            edges = skeleton_data['edges']

            print(f"[UniRigExtractSkeleton] Extracted {len(vertices)} joints, {len(edges)} bones")

            # Normalize to [-1, 1]
            vertices = normalize_skeleton(vertices)
            print(f"[UniRigExtractSkeleton] Normalized to range [{vertices.min():.3f}, {vertices.max():.3f}]")

            # Build skeleton dict with basic data
            skeleton = {
                "vertices": vertices,
                "edges": edges,
            }

            # Add hierarchy data if available (for animation-ready export)
            if 'bone_names' in skeleton_data:
                skeleton['bone_names'] = skeleton_data['bone_names'].tolist()
                skeleton['bone_parents'] = skeleton_data['bone_parents'].tolist()
                skeleton['bone_to_head_vertex'] = skeleton_data['bone_to_head_vertex'].tolist()
                print(f"[UniRigExtractSkeleton] Included hierarchy: {len(skeleton['bone_names'])} bones with parent relationships")
            else:
                print(f"[UniRigExtractSkeleton] No hierarchy data in skeleton (edges-only mode)")

            print(f"[UniRigExtractSkeleton] ✓✓✓ Skeleton extraction complete! ✓✓✓")
            return (skeleton,)

    def _extract_bones_from_fbx(self, fbx_mesh):
        """
        Extract bone structure from FBX.

        FBX armature structure is complex. For now, we extract:
        - Joint positions from mesh vertices
        - Bone connections from edge structure
        """
        # If the FBX has a scene graph with bones, extract from there
        # For now, simplified: use mesh structure as proxy

        if hasattr(fbx_mesh, 'vertices'):
            vertices = np.array(fbx_mesh.vertices)

            # Try to extract edges if available
            if hasattr(fbx_mesh, 'edges'):
                edges = np.array(fbx_mesh.edges)
            elif hasattr(fbx_mesh, 'faces') and len(fbx_mesh.faces) > 0:
                # Extract edges from faces
                faces = fbx_mesh.faces
                edges_set = set()
                for face in faces:
                    for i in range(len(face)):
                        edge = tuple(sorted([face[i], face[(i+1) % len(face)]]))
                        edges_set.add(edge)
                edges = np.array(list(edges_set))
            else:
                # Create minimal spanning tree from vertices
                from scipy.spatial import cKDTree
                tree = cKDTree(vertices)
                edges = []
                for i in range(len(vertices) - 1):
                    # Connect to nearest unconnected neighbor
                    dists, indices = tree.query(vertices[i], k=2)
                    edges.append([i, indices[1]])
                edges = np.array(edges)
        else:
            raise ValueError("Cannot extract bones from FBX: no vertices found")

        return vertices, edges


class UniRigExtractRig:
    """
    Extract full rig (skeleton + skinning weights) using UniRig.

    This node runs both skeleton and skinning prediction.
    Output includes skinning weights for animation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("rigged_mesh",)
    FUNCTION = "extract"
    CATEGORY = "UniRig"

    def extract(self, trimesh, seed):
        """Extract full rig with skinning weights."""
        print(f"[UniRigExtractRig] Starting full rig extraction...")

        if not os.path.exists(UNIRIG_PATH):
            raise RuntimeError(f"UniRig not found at {UNIRIG_PATH}")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.glb")
            skeleton_path = os.path.join(tmpdir, "skeleton.fbx")
            rigged_path = os.path.join(tmpdir, "rigged.fbx")

            # Export mesh
            trimesh.export(input_path)

            # Step 1: Generate skeleton
            print(f"[UniRigExtractRig] Step 1: Generating skeleton...")
            subprocess.run([
                "bash",
                os.path.join(UNIRIG_PATH, "launch", "inference", "generate_skeleton.sh"),
                "--input", input_path,
                "--output", skeleton_path,
                "--seed", str(seed),
            ], cwd=UNIRIG_PATH, check=True, timeout=300)

            if not os.path.exists(skeleton_path):
                raise RuntimeError(f"Skeleton generation failed: {skeleton_path} not created")

            # Step 2: Generate skinning
            print(f"[UniRigExtractRig] Step 2: Generating skinning weights...")
            subprocess.run([
                "bash",
                os.path.join(UNIRIG_PATH, "launch", "inference", "generate_skin.sh"),
                "--input", skeleton_path,
                "--output", rigged_path,
            ], cwd=UNIRIG_PATH, check=True, timeout=300)

            if not os.path.exists(rigged_path):
                raise RuntimeError(f"Skinning generation failed: {rigged_path} not created")

            # Load rigged mesh
            rigged_mesh = trimesh.load(rigged_path)

            print(f"[UniRigExtractRig] Rig extraction complete!")

            return (rigged_mesh,)


class UniRigSaveSkeleton:
    """
    Save skeleton to file in various formats.

    Supports:
    - OBJ, PLY: Simple line mesh visualization
    - FBX, GLB: Full hierarchy (if available) for animation
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton": ("SKELETON",),
                "filename": ("STRING", {"default": "skeleton.fbx"}),
                "format": (["fbx", "glb", "obj", "ply"], {"default": "fbx"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "UniRig"

    def save(self, skeleton, filename, format):
        """Save skeleton to file."""
        print(f"[UniRigSaveSkeleton] Saving skeleton to {filename} as {format.upper()}...")

        # Get ComfyUI output directory
        output_dir = folder_paths.get_output_directory()
        filepath = os.path.join(output_dir, filename)

        # Ensure filename has correct extension
        if not filepath.endswith(f'.{format}'):
            filepath = os.path.splitext(filepath)[0] + f'.{format}'

        vertices = skeleton['vertices']
        edges = skeleton['edges']

        has_hierarchy = 'bone_names' in skeleton and 'bone_parents' in skeleton

        if format in ['fbx', 'glb']:
            if not has_hierarchy:
                print(f"[UniRigSaveSkeleton] Warning: Skeleton has no hierarchy data. FBX/GLB will only contain line mesh.")
                print(f"[UniRigSaveSkeleton] For full animation support, ensure the skeleton was extracted with UniRig.")
                self._save_line_mesh(vertices, edges, filepath, format)
            else:
                self._save_fbx_with_hierarchy(skeleton, filepath, format)
        else:
            # OBJ/PLY: save as line mesh
            self._save_line_mesh(vertices, edges, filepath, format)

        print(f"[UniRigSaveSkeleton] ✓ Saved to: {filepath}")
        return {}

    def _save_line_mesh(self, vertices, edges, filepath, format):
        """Save skeleton as a simple line mesh (OBJ or PLY)."""
        import trimesh

        # Create line segments from edges
        # For trimesh, we create a Path3D object
        entities = []
        for edge in edges:
            entities.append(trimesh.path.entities.Line([edge[0], edge[1]]))

        path = trimesh.path.Path3D(
            vertices=vertices,
            entities=entities
        )

        # Export
        path.export(filepath, file_type=format)

    def _save_fbx_with_hierarchy(self, skeleton, filepath, format):
        """Save skeleton with full hierarchy using Blender."""
        import pickle
        import tempfile

        vertices = skeleton['vertices']
        edges = skeleton['edges']
        bone_names = skeleton['bone_names']
        bone_parents = skeleton['bone_parents']
        bone_to_head_vertex = skeleton['bone_to_head_vertex']

        # Denormalize vertices (they're in [-1, 1] range)
        # Scale up to reasonable size (e.g., 1 meter = 1 unit)
        vertices_denorm = vertices.copy()

        # Reconstruct joint positions for each bone
        # bone_to_head_vertex maps bone index to the vertex index of its head
        joints = []
        for bone_idx in range(len(bone_names)):
            head_vertex_idx = bone_to_head_vertex[bone_idx]
            joints.append(vertices_denorm[head_vertex_idx])

        joints = np.array(joints, dtype=np.float32)

        # Prepare data for Blender export (plain Python types for pickle)
        data = {
            'joints': joints.tolist(),
            'parents': bone_parents,
            'names': bone_names,
            'vertices': None,  # No mesh
            'faces': None,
            'skin': None,
            'tails': None,  # Will be auto-calculated
            'group_per_vertex': -1,
            'do_not_normalize': True,
        }

        # Save to temporary pickle file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle_path = f.name
            pickle.dump(data, f)

        try:
            # Find Blender executable and wrapper script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            wrapper_script = os.path.join(current_dir, 'lib', 'blender_export_fbx.py')

            if not os.path.exists(wrapper_script):
                raise RuntimeError(f"Blender export script not found: {wrapper_script}")

            if not os.path.exists(BLENDER_EXE):
                raise RuntimeError(f"Blender executable not found: {BLENDER_EXE}")

            # Determine output format
            if format == 'glb':
                # For GLB, first export to FBX, then convert
                temp_fbx = filepath.replace('.glb', '_temp.fbx')
                output_path = temp_fbx
            else:
                output_path = filepath

            # Build Blender command
            cmd = [
                BLENDER_EXE,
                '--background',
                '--python', wrapper_script,
                '--',
                pickle_path,
                output_path,
                '--extrude_size=0.03',
            ]

            # Run Blender
            print(f"[UniRigSaveSkeleton] Running Blender to export {format.upper()}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                print(f"[UniRigSaveSkeleton] Blender error: {result.stderr}")
                raise RuntimeError(f"FBX export failed with return code {result.returncode}")

            if not os.path.exists(output_path):
                raise RuntimeError(f"Export completed but output file not found: {output_path}")

            # Convert FBX to GLB if needed
            if format == 'glb':
                print(f"[UniRigSaveSkeleton] Converting FBX to GLB...")
                import trimesh
                mesh = trimesh.load(temp_fbx)
                mesh.export(filepath)
                os.remove(temp_fbx)

        finally:
            # Clean up pickle file
            if os.path.exists(pickle_path):
                os.unlink(pickle_path)


NODE_CLASS_MAPPINGS = {
    "UniRigExtractSkeleton": UniRigExtractSkeleton,
    "UniRigExtractRig": UniRigExtractRig,
    "UniRigSaveSkeleton": UniRigSaveSkeleton,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniRigExtractSkeleton": "UniRig: Extract Skeleton",
    "UniRigExtractRig": "UniRig: Extract Full Rig",
    "UniRigSaveSkeleton": "UniRig: Save Skeleton",
}

#!/usr/bin/env python
# coding=utf-8
"""
HDF5 to LeRobot Compatible Format Converter

This script converts HDF5 files in the subtasks directory to the LeRobot compatible format
required by Isaac-GR00T.

Usage:
    python hdf5_to_lerobot_converter.py --input-dir /Users/lr-2002/test_dataset/subtasks --output-dir /Users/lr-2002/lerobot_dataset

Author: Cascade AI
Date: 2025-06-23
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import h5py
import imageio
import cv2
from tqdm import tqdm
import shutil
from multiprocessing import Pool
from pathlib import Path

# 最大进程数限制
MAX_PROCESSES = 4


def decode_image(image_lists):
    padded_compressed_image_list = image_lists
    image_list = []
    for padded_compressed_image in padded_compressed_image_list:
        image = cv2.imdecode(padded_compressed_image, 1)
        image_list.append(image)
    return image_list


def process_camera_video(params):
    """
    Process and save video for a single camera.
    
    Args:
        params: Tuple containing (image_data, cam_name, cam_dir, episode_name, fps)
    
    Returns:
        Camera name if successful, None otherwise
    """
    image_data, cam_name, cam_dir, episode_name, fps = params
    
    try:
        # Create directory for this camera
        os.makedirs(cam_dir, exist_ok=True)
        video_path = os.path.join(cam_dir, f"{episode_name}.mp4")
        
        print(f"Processing camera {cam_name}, data shape: {image_data.shape}")
        
        # Process the image data if it's flat (needs decoding)
        if len(image_data.shape) == 2:  # Flattened image data
            # Decode the compressed images
            image_data = decode_image(image_data)
            print(f"after change image data  {len(image_data)}, {image_data[0].shape}")
        
        # Save the video
        print(f"Saving video: {video_path}")
        with imageio.get_writer(
            video_path, format="ffmpeg", fps=fps, codec="libx264", quality=8
        ) as writer:
            for frame in tqdm(image_data, desc=f"Writing {cam_name}"):
                writer.append_data(frame)
                
        print(f"[✓] Saved video to: {video_path}")
        return cam_name
    
    except Exception as e:
        print(f"Error saving video for {cam_name}: {e}")
        return None


def save_camera_videos(hdf5_file, output_dir, episode_name, fps=30):
    """
    Extract and save camera videos from an HDF5 file sequentially.

    Args:
        hdf5_file: Open h5py.File object
        output_dir: Base output directory
        episode_name: Episode name for file naming
        fps: Frames per second for the video

    Returns:
        List of camera names that were successfully saved
    """
    video_dir = os.path.join(output_dir, "videos", "chunk-000")
    camera_names = []

    # Check if observations/images exists
    if "observations/images" not in hdf5_file:
        print("No camera data found in observations/images")
        return camera_names
    
    # 顺序处理每个摄像头（不在子进程中创建新的进程池）
    for cam_name in hdf5_file["observations/images"]:
        if "cam" not in cam_name.lower():
            continue
            
        # Get the image data for this camera
        image_data = hdf5_file[f"observations/images/{cam_name}"][()] 
        cam_dir = os.path.join(video_dir, f"observation.images.{cam_name}")
        
        # 直接处理单个摄像头
        result = process_camera_video((image_data, cam_name, cam_dir, episode_name, fps))
        if result:
            camera_names.append(result)
                
    return camera_names


def create_metadata(output_dir, dataset_info, modality_config):
    """
    Create the required metadata files.

    Args:
        output_dir: Output directory
        dataset_info: Dataset information
        modality_config: Modality configuration
    """
    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    # Create modality.json
    with open(os.path.join(meta_dir, "modality.json"), "w") as f:
        json.dump(modality_config, f, indent=2)

    # Create info.json
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"[✓] Created metadata files in {meta_dir}")


def create_episodes_tasks(output_dir, episodes_data, tasks_data):
    """
    Create episodes.jsonl and tasks.jsonl files.

    Args:
        output_dir: Output directory
        episodes_data: List of episode information dictionaries
        tasks_data: List of task information dictionaries
    """
    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    # Create episodes.jsonl
    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for episode in episodes_data:
            f.write(json.dumps(episode) + "\n")

    # Create tasks.jsonl
    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        for task in tasks_data:
            f.write(json.dumps(task) + "\n")

    print(f"[✓] Created episodes.jsonl and tasks.jsonl in {meta_dir}")


def process_hdf5_file(file_path, output_dir, episode_index):
    """
    Process a single HDF5 file into the LeRobot compatible format.

    Args:
        file_path: Path to HDF5 file
        output_dir: Output directory
        episode_index: Episode index for naming

    Returns:
        episode_info: Episode information dictionary
    """
    print(f"\nProcessing HDF5 file: {file_path}")

    with h5py.File(file_path, "r") as f:
        # Print structure
        print("File structure:")

        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")

        f.visititems(print_structure)

        # Create directory structure
        data_dir = os.path.join(output_dir, "data", "chunk-000")
        os.makedirs(data_dir, exist_ok=True)

        # Process episode name
        episode_name = f"episode_{episode_index:06d}"

        # Save camera videos using the new function
        # This handles the directory creation and proper reshaping
        camera_names = save_camera_videos(f, output_dir, episode_name)

        # 2. Process state and action data
        state_data = {}
        action_data = {}

        # Get episode length from camera data if available
        episode_length = 0
        if "observations/images" in f:
            for cam_name in camera_names:
                path = f"observations/images/{cam_name}"
                if path in f:
                    episode_length = max(episode_length, f[path].shape[0])

        if "observations" in f:
            if "qpos" in f["observations"]:
                state_data["qpos"] = f["observations/qpos"][()]
                episode_length = max(episode_length, state_data["qpos"].shape[0])
            if "qvel" in f["observations"]:
                state_data["qvel"] = f["observations/qvel"][()]
                episode_length = max(episode_length, state_data["qvel"].shape[0])
            if "ee" in f["observations"]:
                state_data["ee"] = f["observations/ee"][()]
                episode_length = max(episode_length, state_data["ee"].shape[0])
            if "torque" in f["observations"]:
                state_data["torque"] = f["observations/torque"][()]
                episode_length = max(episode_length, state_data["torque"].shape[0])

        if "action" in f:  
            action_data["action"] = f["action"][()] 
            episode_length = max(episode_length, action_data["action"].shape[0])

        # 3. Save episode data to parquet
        if state_data or action_data:
            # Convert arrays to DataFrame for parquet
            # Build a list of dictionaries, where each dict represents a row
            rows = []

            for i in range(episode_length):
                row = {}

                # Add state columns
                for key, value in state_data.items():
                    if i < value.shape[0]:
                        if len(value.shape) > 1 and value.shape[1] > 1:
                            # For multi-dimensional data
                            for j in range(value.shape[1]):
                                row[f"{key}_{j}"] = value[i, j]
                        else:
                            # For 1D data
                            row[key] = value[i][0] if len(value.shape) > 1 else value[i]
                    else:
                        # Fill with nan if index out of bounds
                        if key not in state_data:
                            row[key] = np.nan

                # Add action columns
                for key, value in action_data.items():
                    if i < value.shape[0]:
                        if len(value.shape) > 1 and value.shape[1] > 1:
                            # For multi-dimensional data
                            for j in range(value.shape[1]):
                                row[f"{key}_{j}"] = value[i, j]
                        else:
                            # For 1D data
                            row[key] = value[i][0] if len(value.shape) > 1 else value[i]
                    else:
                        # Fill with nan if index out of bounds
                        # This condition was incorrect (key is always in action_data here)
                        # Just fill with nan when index is out of bounds
                        if len(value.shape) > 1 and value.shape[1] > 1:
                            for j in range(value.shape[1]):
                                row[f"{key}_{j}"] = np.nan
                        else:
                            row[key] = np.nan

                rows.append(row)

            # Create DataFrame
            df = pd.DataFrame(rows)

            # Save to parquet
            parquet_path = os.path.join(data_dir, f"{episode_name}.parquet")
            df.to_parquet(parquet_path, index=False)
            print(f"[✓] Saved parquet to: {parquet_path}")

            # Build modality config
            modality_config = {"state": {}, "action": {}, "video": {}}

            # Add state modalities with left/right arm split
            state_offset = 0
            
            # Process state data with left/right arm division
            for key, value in state_data.items():
                if len(value.shape) > 1 and value.shape[1] >= 14:  # Contains both arms (assuming 7+7 structure)
                    # Left arm (first 7 values)
                    modality_config["state"][f"{key}_left_arm"] = {
                        "start": state_offset,
                        "end": state_offset + 7
                    }
                    state_offset += 7
                    
                    # Right arm (next 7 values)
                    modality_config["state"][f"{key}_right_arm"] = {
                        "start": state_offset,
                        "end": state_offset + 7
                    }
                    state_offset += 7
                    
                    # Any remaining values
                    if value.shape[1] > 14:
                        extra_dims = value.shape[1] - 14
                        modality_config["state"][f"{key}_extra"] = {
                            "start": state_offset,
                            "end": state_offset + extra_dims
                        }
                        state_offset += extra_dims
                else:
                    # For states that don't follow the arm split pattern
                    dim = value.shape[1] if len(value.shape) > 1 else 1
                    modality_config["state"][key] = {
                        "start": state_offset,
                        "end": state_offset + dim,
                    }
                    state_offset += dim
            
            # Add action modalities with left/right arm split
            action_offset = 0
            
            # Process action data with left/right arm division
            for key, value in action_data.items():
                if len(value.shape) > 1 and value.shape[1] >= 14:  # Contains both arms
                    # Left arm (first 7 values)
                    modality_config["action"][f"{key}_left_arm"] = {
                        "start": action_offset,
                        "end": action_offset + 7
                    }
                    action_offset += 7
                    
                    # Right arm (next 7 values)
                    modality_config["action"][f"{key}_right_arm"] = {
                        "start": action_offset,
                        "end": action_offset + 7
                    }
                    action_offset += 7
                    
                    # Any remaining values
                    if value.shape[1] > 14:
                        extra_dims = value.shape[1] - 14
                        modality_config["action"][f"{key}_extra"] = {
                            "start": action_offset,
                            "end": action_offset + extra_dims
                        }
                        action_offset += extra_dims
                else:
                    # For actions that don't follow the arm split pattern
                    dim = value.shape[1] if len(value.shape) > 1 else 1
                    modality_config["action"][key] = {
                        "start": action_offset,
                        "end": action_offset + dim,
                    }
                    action_offset += dim

            # Add video modalities
            for cam_name in camera_names:
                modality_config["video"][cam_name] = {
                    "original_key": f"observations/images/{cam_name}"
                }

            # Return episode info
            return (
                {
                    "episode_index": episode_index,
                    "tasks": [
                        "Fold the clothes and put it to the center of the desk"
                    ],  # Empty for now, could be populated if task data is available
                    "length": episode_length,
                },
                modality_config,
            )

    return None, None


def process_file_task(params):
    """
    Process a single HDF5 file in a separate process.
    
    Args:
        params: Tuple containing (file_path, output_dir, idx)
    
    Returns:
        Tuple containing (episode_info, file_modality_config)
    """
    file_path, output_dir, idx = params
    
    try:
        # 重要：在子进程中不要尝试再创建进程池
        episode_info, file_modality_config = process_hdf5_file(file_path, output_dir, idx)
        print(f"\n✅ Successfully processed: {os.path.basename(file_path)}")
        return episode_info, file_modality_config
    except Exception as e:
        print(f"\n❌ Error processing {os.path.basename(file_path)}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main(input_dir, output_dir):
    """
    Main function to convert HDF5 files to LeRobot compatible format.

    Args:
        input_dir: Input directory with HDF5 files
        output_dir: Output directory for LeRobot compatible format
    """
    print(f"Converting HDF5 files from {input_dir} to LeRobot format in {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all HDF5 files
    hdf5_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".hdf5"):
                hdf5_files.append(os.path.join(root, file))

    if not hdf5_files:
        print(f"No HDF5 files found in {input_dir}")
        return

    print(f"Found {len(hdf5_files)} HDF5 files")

    # Prepare parameters for parallel processing
    params = [(file_path, output_dir, idx) for idx, file_path in enumerate(hdf5_files)]
    
    # Process files in parallel
    episodes_data = []
    tasks_data = [{"task_index": 0, "task": "robot manipulation task"}]  # Default task
    modality_config = None
    
    # Use multiprocessing to process files in parallel
    num_processes = min(MAX_PROCESSES, os.cpu_count(), len(hdf5_files))
    print(f"Using {num_processes} processes for parallel processing")
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_file_task, params)
        
        # Collect results
        for idx, (episode_info, file_modality_config) in enumerate(results):
            if episode_info:
                episodes_data.append(episode_info)
                
            if idx == 0 and file_modality_config:
                # Use the first file's modality config as template
                modality_config = file_modality_config

    # Create dataset information
    dataset_info = {
        "name": "Converted Robot Dataset",
        "description": "Robot manipulation dataset converted from HDF5 format",
        "version": "1.0",
        "author": "Converted by HDF5 to LeRobot Converter",
        "date": "2025-06-23",
        "license": "Proprietary",
    }

    # Create metadata files
    if modality_config:
        create_metadata(output_dir, dataset_info, modality_config)
    else:
        print("Warning: No modality configuration could be extracted from files")

    # Create episodes and tasks files
    create_episodes_tasks(output_dir, episodes_data, tasks_data)

    print("\n[✓] Conversion completed successfully!")
    print(f"    Output directory: {output_dir}")
    print(f"    Episodes processed: {len(episodes_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert HDF5 files to LeRobot compatible format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/Users/lr-2002/test_dataset/subtasks",
        help="Input directory with HDF5 files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/lr-2002/lerobot_dataset",
        help="Output directory for LeRobot compatible format",
    )

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)

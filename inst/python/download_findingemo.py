#!/usr/bin/env python3
"""
Download FindingEmo-Light dataset with proper flat structure and image limiting.

This script provides a function-based approach for downloading the FindingEmo dataset
that integrates properly with R via reticulate, similar to image.py.

Key features:
- True flat directory structure (all images in images/ folder)  
- Proper max_images limiting DURING download, not after
- Randomization support
- Progress feedback
- Robust error handling

Author: transforEmotion team
License: Same as transforEmotion package
"""

import os
import json
import random
import shutil
from pathlib import Path
from urllib import request
import requests
from termcolor import cprint
import pandas as pd


def _download_img(url: str, file_path: str, b_test=False):
    """
    Download an image from URL. Based on findingemo_light download_multi.
    
    Args:
        url: URL to the image
        file_path: Full path where image will be saved
        b_test: If True, just test URL without downloading
    """
    req = request.Request(
        url=url,
        headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
        }
    )
    res = request.urlopen(req, timeout=10)

    if not b_test:
        # Create output directory if needed
        file_dir = os.path.dirname(file_path)
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        # Download image
        with open(file_path, 'wb') as fout:
            fout.write(res.read())


def _query_waybackmachine(url: str, file_path: str, b_test=False):
    """
    Try to download from WayBackMachine. Based on findingemo_light download_multi.
    
    Args:
        url: Original URL to look up
        file_path: Where to save the image
        b_test: If True, just test without downloading
        
    Returns:
        1 if successful, 0 if not found, -1 if error
    """
    payload = {'url': url}
    archive_query_url = 'https://archive.org/wayback/available'
    
    try:
        response = requests.post(archive_query_url, data=payload, timeout=10, verify=True)
        archived_results = json.loads(response.content)
        archived_results = archived_results['results'][0]['archived_snapshots']
    except:
        return -1

    if archived_results:
        archived_url = archived_results['closest']['url']
        try:
            # Convert URL to direct link to image
            idx = archived_url.find(url)
            if idx < 0 and url.startswith('http://'):
                idx = archived_url.find(url.replace('http://', 'https://'))
            archived_url = archived_url[:idx - 1] + "if_" + archived_url[idx - 1:]
            
            _download_img(url=archived_url, file_path=file_path, b_test=b_test)
            print("Image downloaded through WayBackMachine!")
            return 1
        except Exception as e:
            print(f"Found WayBackMachine match but couldn't download: {e}")
            return -1
    else:
        return 0


def download_findingemo_data(target_dir, max_images=None, randomize=False, 
                           skip_existing=True, force=False):
    """
    Download FindingEmo-Light dataset with flat structure and proper image limiting.
    
    Args:
        target_dir (str): Directory to download images to
        max_images (int, optional): Maximum number of images to download
        randomize (bool): If True and max_images is set, randomly select images
        skip_existing (bool): Skip download if dataset already exists
        force (bool): Force download even if dataset exists
    
    Returns:
        dict: Status information with success flag and details
    """
    try:
        # Import findingemo-light to get access to their data
        try:
            import findingemo_light
            # Get the path to their dataset_urls_exploded.json file
            abs_dir = os.path.dirname(os.path.abspath(findingemo_light.__file__))
            url_file = os.path.join(abs_dir, 'data', 'dataset_urls_exploded.json')
            annotation_file = os.path.join(abs_dir, 'data', 'annotations_single.ann')
        except ImportError as e:
            return {
                "success": False,
                "message": f"Failed to import findingemo_light. Please install using setup_modules(): {e}",
                "error_type": "ImportError"
            }
        
        # Create target directory
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Handle force flag
        if force:
            skip_existing = False
        
        # Check if dataset already exists
        if skip_existing:
            annotation_output = target_path / "annotations.csv"
            urls_output = target_path / "urls.json"
            if annotation_output.exists() and urls_output.exists():
                return {
                    "success": True,
                    "message": "Dataset already exists",
                    "target_dir": str(target_path),
                    "skipped": True,
                    "annotation_file": str(annotation_output),
                    "urls_file": str(urls_output)
                }
        
        print(f"Downloading FindingEmo dataset to: {target_dir}")
        if max_images:
            print(f"Limiting to {max_images} images" + 
                  (" (randomly selected)" if randomize else " (first N)"))
        
        # Load the URLs data
        if not os.path.exists(url_file):
            return {
                "success": False,
                "message": f"URLs file not found: {url_file}"
            }
            
        with open(url_file, 'r') as f:
            json_data = json.load(f)
        
        print(f"Total available images: {len(json_data)}")
        
        # Apply randomization if requested
        if randomize and max_images is not None:
            random.seed(42)  # For reproducibility
            random.shuffle(json_data)
            print("Randomized image order")
        
        # Limit the data if max_images is specified
        if max_images is not None and len(json_data) > max_images:
            max_images = int(max_images)  # Convert to int for slicing
            json_data = json_data[:max_images]
            print(f"Limited to first {len(json_data)} images")
        
        # Create flat directory structure
        images_dir = target_path / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Download images with the same logic as findingemo_light
        downloaded_imgs = set()
        not_found = set()
        
        print(f"Downloading {len(json_data)} images...")
        
        for img_idx, img_data in enumerate(json_data):
            rel_path = img_data['rel_path']
            img_url = img_data['url']
            url_idx = img_data['idx_url']
            
            # Create flat filename from relative path (remove directories, keep extension)
            flat_filename = os.path.basename(rel_path)
            
            img_path = images_dir / flat_filename
            
            # Skip if already downloaded
            if str(img_path) in downloaded_imgs or (not force and img_path.exists()):
                if str(img_path) not in downloaded_imgs:
                    print(f"Image [{flat_filename}] already exists, skipping...")
                    downloaded_imgs.add(str(img_path))
                continue
            
            print(f"\rTrying URL {url_idx}... ({img_idx+1}/{len(json_data)})", end='', flush=True)
            
            # Download image
            try:
                _download_img(url=img_url, file_path=str(img_path), b_test=False)
                downloaded_imgs.add(str(img_path))
                print(f" Downloaded: [{img_url}]")
                if rel_path in not_found:
                    not_found.remove(rel_path)
                    
            except Exception as e:
                print(f'\nCould not download image: {rel_path}')
                print(f"URL: {img_url}")
                print(f"Error: {e}")
                print("Trying WayBackMachine...", end='', flush=True)
                
                wayback_hit = _query_waybackmachine(img_url, str(img_path), b_test=False)
                if wayback_hit < 1:
                    not_found.add(rel_path)
                    if wayback_hit == 0:
                        print(" no potatoes.")
                else:
                    downloaded_imgs.add(str(img_path))
                    if rel_path in not_found:
                        not_found.remove(rel_path)
            
            print(f"Downloaded {len(downloaded_imgs)} images. Not found: {len(not_found)}")
        
        print(f"\nSuccessfully downloaded {len(downloaded_imgs)} images")
        if not_found:
            print(f"Failed to download {len(not_found)} images")
        
        # Load and filter annotations to match downloaded images
        annotations_data = None
        if os.path.exists(annotation_file):
            try:
                annotations_data = pd.read_csv(annotation_file)
                print(f"Loaded {len(annotations_data)} annotations")
                
                # Filter annotations to match selected images
                selected_rel_paths = [img_data['rel_path'] for img_data in json_data]
                
                # Find the image ID column
                id_columns = ['image_id', 'id', 'Image_ID', 'ID']
                id_column = None
                for col in id_columns:
                    if col in annotations_data.columns:
                        id_column = col
                        break
                
                if id_column:
                    # The rel_path contains the image ID typically
                    selected_ids = []
                    for rel_path in selected_rel_paths:
                        # Extract ID from rel_path (typically the filename without extension)
                        img_id = os.path.splitext(os.path.basename(rel_path))[0]
                        selected_ids.append(img_id)
                    
                    initial_count = len(annotations_data)
                    annotations_data = annotations_data[
                        annotations_data[id_column].astype(str).isin(selected_ids)
                    ]
                    final_count = len(annotations_data)
                    print(f"Filtered annotations: {initial_count} → {final_count}")
                    
            except Exception as e:
                print(f"Warning: Could not load/filter annotations: {e}")
                annotations_data = None
        
        # Save filtered URLs data
        urls_output_file = target_path / "urls.json"
        with open(urls_output_file, 'w') as f:
            # Convert to the same format as before for compatibility
            urls_dict = {}
            for img_data in json_data:
                # Use relative path as key for compatibility
                key = os.path.splitext(os.path.basename(img_data['rel_path']))[0]
                urls_dict[key] = img_data['url']
            json.dump(urls_dict, f, indent=2)
        
        # Save filtered annotations
        annotations_output_file = target_path / "annotations.csv"
        if annotations_data is not None:
            annotations_data.to_csv(annotations_output_file, index=False)
        
        # Create metadata file
        metadata = {
            "dataset": "FindingEmo-Light",
            "structure": "flat",
            "total_available_images": len(json_data) if max_images is None else len(json_data) + (len(json_data) if max_images else 0),
            "selected_images": len(json_data),
            "downloaded_images": len(downloaded_imgs),
            "failed_downloads": len(not_found),
            "max_images": max_images,
            "randomized": randomize and max_images is not None,
            "files": {
                "annotations": "annotations.csv",
                "urls": "urls.json",
                "images": "images/"
            }
        }
        
        metadata_file = target_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✓ Dataset download completed successfully")
        
        # Prepare success response
        success_message = "Dataset downloaded successfully"
        if len(downloaded_imgs) == 0:
            success_message += " (metadata only - no images downloaded)"
        elif max_images and len(downloaded_imgs) < max_images:
            success_message += f" (partial: {len(downloaded_imgs)}/{max_images} images downloaded)"
        
        return {
            "success": True,
            "message": success_message,
            "target_dir": str(target_path),
            "annotation_file": str(annotations_output_file),
            "urls_file": str(urls_output_file),
            "metadata_file": str(metadata_file),
            "image_count": len(downloaded_imgs),
            "failed_count": len(not_found),
            "total_selected": len(json_data),
            "skipped": False
        }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Download failed: {str(e)}",
            "error_type": type(e).__name__
        }


def check_findingemo_dependencies():
    """
    Check if required dependencies are available.
    
    Returns:
        dict: Status of dependencies
    """
    dependencies = {}
    
    try:
        import requests
        dependencies['requests'] = True
    except ImportError:
        dependencies['requests'] = False
    
    try:
        import pandas
        dependencies['pandas'] = True
    except ImportError:
        dependencies['pandas'] = False
        
    try:
        import findingemo_light
        dependencies['findingemo_light'] = True
    except ImportError:
        dependencies['findingemo_light'] = False
    
    try:
        from termcolor import cprint
        dependencies['termcolor'] = True
    except ImportError:
        dependencies['termcolor'] = False
    
    all_available = all(dependencies.values())
    
    return {
        "all_available": all_available,
        "dependencies": dependencies,
        "missing": [dep for dep, available in dependencies.items() if not available]
    }

#!/usr/bin/env python3
"""
Load and preprocess FindingEmo-Light dataset annotations.

This script handles loading the CSV annotations from the FindingEmo dataset,
performs basic preprocessing and validation, and outputs clean data for R consumption.

Usage:
    python load_findingemo_annotations.py --data_dir /path/to/findingemo --output_format json

Author: transforEmotion team
License: Same as transforEmotion package
"""

import argparse
import json
import os
import sys
import pandas as pd
from pathlib import Path


def load_annotations(data_dir, output_format='json'):
    """
    Load and preprocess FindingEmo annotations.
    
    Args:
        data_dir (str): Directory containing the downloaded FindingEmo data
        output_format (str): Output format ('json' or 'csv')
    
    Returns:
        dict: Status information with success flag and processed data
    """
    try:
        # Expected annotation file path
        annotation_file = os.path.join(data_dir, "data", "annotations_single.ann")
        urls_file = os.path.join(data_dir, "data", "dataset_urls_exploded.json")
        
        # Check if files exist
        if not os.path.exists(annotation_file):
            return {
                "success": False,
                "message": f"Annotation file not found: {annotation_file}",
                "error_type": "FileNotFoundError"
            }
        
        if not os.path.exists(urls_file):
            return {
                "success": False,
                "message": f"URLs file not found: {urls_file}",
                "error_type": "FileNotFoundError"
            }
        
        print(f"Loading annotations from: {annotation_file}")
        
        # Load annotations CSV
        try:
            annotations_df = pd.read_csv(annotation_file)
            print(f"✓ Loaded {len(annotations_df)} annotations")
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to read annotations CSV: {e}",
                "error_type": "CSVReadError"
            }
        
        # Load URLs JSON
        try:
            with open(urls_file, 'r') as f:
                urls_data = json.load(f)
            print(f"✓ Loaded URL data with {len(urls_data)} entries")
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to read URLs JSON: {e}",
                "error_type": "JSONReadError"
            }
        
        # Basic preprocessing and validation
        print("Preprocessing annotations...")
        
        # Check for expected columns
        expected_cols = ['image_id', 'valence', 'arousal']  # Adjust based on actual schema
        missing_cols = [col for col in expected_cols if col not in annotations_df.columns]
        if missing_cols:
            print(f"Warning: Missing expected columns: {missing_cols}")
            print(f"Available columns: {list(annotations_df.columns)}")
        
        # Remove rows with missing critical data
        initial_count = len(annotations_df)
        annotations_df = annotations_df.dropna(subset=[col for col in expected_cols if col in annotations_df.columns])
        final_count = len(annotations_df)
        
        if initial_count > final_count:
            print(f"Removed {initial_count - final_count} rows with missing data")
        
        # Validate valence and arousal ranges (typically -1 to 1 or 0 to 1)
        if 'valence' in annotations_df.columns:
            valence_stats = {
                'min': float(annotations_df['valence'].min()),
                'max': float(annotations_df['valence'].max()),
                'mean': float(annotations_df['valence'].mean())
            }
            print(f"Valence range: {valence_stats['min']:.3f} to {valence_stats['max']:.3f}")
        
        if 'arousal' in annotations_df.columns:
            arousal_stats = {
                'min': float(annotations_df['arousal'].min()),
                'max': float(annotations_df['arousal'].max()),
                'mean': float(annotations_df['arousal'].mean())
            }
            print(f"Arousal range: {arousal_stats['min']:.3f} to {arousal_stats['max']:.3f}")
        
        # Convert emotion labels if present
        emotion_mapping = None
        if 'emotion_label' in annotations_df.columns:
            unique_emotions = annotations_df['emotion_label'].unique()
            emotion_mapping = {emotion: i for i, emotion in enumerate(sorted(unique_emotions))}
            print(f"Found emotions: {list(unique_emotions)}")
        
        # Prepare output data
        processed_data = {
            "annotations": annotations_df.to_dict('records'),
            "urls": urls_data,
            "metadata": {
                "n_annotations": len(annotations_df),
                "n_urls": len(urls_data),
                "columns": list(annotations_df.columns),
                "emotion_mapping": emotion_mapping
            }
        }
        
        if 'valence' in annotations_df.columns:
            processed_data["metadata"]["valence_stats"] = valence_stats
        
        if 'arousal' in annotations_df.columns:
            processed_data["metadata"]["arousal_stats"] = arousal_stats
        
        return {
            "success": True,
            "message": "Annotations loaded and preprocessed successfully",
            "data": processed_data,
            "output_format": output_format
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to load annotations: {e}",
            "error_type": type(e).__name__
        }


def save_processed_data(result, output_file=None):
    """
    Save processed data to file.
    
    Args:
        result (dict): Result from load_annotations
        output_file (str, optional): Output file path
    
    Returns:
        str: Path to saved file
    """
    if not result['success']:
        return None
    
    data = result['data']
    output_format = result['output_format']
    
    if output_file is None:
        output_file = f"findingemo_processed.{output_format}"
    
    try:
        if output_format == 'json':
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        elif output_format == 'csv':
            # Save annotations as CSV
            annotations_df = pd.DataFrame(data['annotations'])
            output_file = output_file.replace('.csv', '_annotations.csv')
            annotations_df.to_csv(output_file, index=False)
            
            # Save metadata as JSON
            metadata_file = output_file.replace('_annotations.csv', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump({
                    'metadata': data['metadata'],
                    'urls': data['urls']
                }, f, indent=2)
            
            return [output_file, metadata_file]
        
        return output_file
        
    except Exception as e:
        print(f"Failed to save processed data: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Load and preprocess FindingEmo-Light annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python load_findingemo_annotations.py --data_dir ./findingemo_data
    python load_findingemo_annotations.py --data_dir /tmp/findingemo --output_format csv
        """
    )
    
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing downloaded FindingEmo data"
    )
    
    parser.add_argument(
        "--output_format",
        choices=['json', 'csv'],
        default='json',
        help="Output format for processed data (default: json)"
    )
    
    parser.add_argument(
        "--output_file",
        help="Output file path (optional)"
    )
    
    parser.add_argument(
        "--output_json",
        help="Output results to JSON file for R integration"
    )
    
    args = parser.parse_args()
    
    print("FindingEmo-Light Annotation Loader")
    print("=================================")
    
    # Load and process annotations
    result = load_annotations(
        data_dir=args.data_dir,
        output_format=args.output_format
    )
    
    # Output results
    print("\nResults:")
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    
    if result['success']:
        metadata = result['data']['metadata']
        print(f"Annotations loaded: {metadata['n_annotations']}")
        print(f"URLs loaded: {metadata['n_urls']}")
        print(f"Columns: {', '.join(metadata['columns'])}")
        
        # Save processed data
        saved_file = save_processed_data(result, args.output_file)
        if saved_file:
            if isinstance(saved_file, list):
                print(f"Saved files: {', '.join(saved_file)}")
            else:
                print(f"Saved to: {saved_file}")
    
    # Save results to JSON if requested (for R integration)
    if args.output_json:
        try:
            # Remove large data from result for JSON output
            json_result = {k: v for k, v in result.items() if k != 'data'}
            if result['success']:
                json_result['metadata'] = result['data']['metadata']
            
            with open(args.output_json, 'w') as f:
                json.dump(json_result, f, indent=2)
            print(f"Results saved to: {args.output_json}")
        except Exception as e:
            print(f"Warning: Failed to save JSON output: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()
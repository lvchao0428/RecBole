#!/usr/bin/env python3
"""
Check views.json format

Quick script to inspect the actual format of views.json files.
"""

import json
from pathlib import Path
import sys

def check_views_json(dataset_name):
    """Check the views.json file format."""
    dataset_path = Path("/home/charlie/project/RecBole/dataset") / dataset_name
    views_json_path = dataset_path / "qwen3_4views" / "views.json"
    
    print(f"\n{'='*80}")
    print(f"Checking views.json for {dataset_name}")
    print(f"{'='*80}\n")
    
    print(f"Path: {views_json_path}")
    
    if not views_json_path.exists():
        print(f"✗ File does not exist!")
        return
    
    print(f"✓ File exists\n")
    
    # Read and display content
    with open(views_json_path, 'r') as f:
        content = f.read()
    
    print(f"Raw content ({len(content)} bytes):")
    print("-" * 80)
    print(content)
    print("-" * 80)
    print()
    
    # Parse JSON
    try:
        data = json.loads(content)
        print(f"✓ Valid JSON\n")
        
        print(f"Structure:")
        print(f"  Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"  Keys: {list(data.keys())}")
            print(f"\nDetailed content:")
            for key, value in data.items():
                print(f"  '{key}': {value} (type: {type(value).__name__})")
        elif isinstance(data, list):
            print(f"  Length: {len(data)}")
            print(f"  First item: {data[0] if data else 'N/A'}")
        
        print()
        
        # Try to parse view mapping
        print("Attempting to parse view mapping...")
        view_mapping = {}
        prompt_info = {}
        
        if isinstance(data, dict):
            if "prompts" in data and isinstance(data["prompts"], list):
                print("  Format: Prompts list (with prompt text)")
                for prompt_item in data["prompts"]:
                    idx = prompt_item["index"]
                    prompt_text = prompt_item["prompt"]
                    
                    # Extract a short name from the prompt
                    if "Identify" in prompt_text:
                        name = "identity"
                    elif "function" in prompt_text or "feature" in prompt_text:
                        name = "function"
                    elif "audience" in prompt_text or "user group" in prompt_text:
                        name = "audience"
                    elif "Categorize" in prompt_text or "context" in prompt_text:
                        name = "category"
                    else:
                        name = f"view_{idx}"
                    
                    view_mapping[idx] = name
                    prompt_info[name] = {
                        'prompt': prompt_text,
                        'file': prompt_item.get('file', f'view_{idx}.npy'),
                        'vector_dim': prompt_item.get('vector_dim', 64)
                    }
                    
            elif "views" in data and isinstance(data["views"], list):
                print("  Format: List with 'views' key")
                for view_info in data["views"]:
                    view_mapping[view_info["id"]] = view_info["name"]
            elif "view_names" in data:
                print("  Format: 'view_names' list")
                for idx, name in enumerate(data["view_names"]):
                    view_mapping[idx] = name
            else:
                print("  Format: Direct mapping (filtering numeric keys)")
                for k, v in data.items():
                    try:
                        view_mapping[int(k)] = v
                    except (ValueError, TypeError):
                        print(f"    Skipping non-numeric key: '{k}'")
        
        if view_mapping:
            print(f"\n✓ Successfully parsed {len(view_mapping)} views:")
            for idx, name in sorted(view_mapping.items()):
                print(f"    {idx}: {name}")
                if name in prompt_info:
                    prompt_preview = prompt_info[name]['prompt'][:60] + '...' if len(prompt_info[name]['prompt']) > 60 else prompt_info[name]['prompt']
                    print(f"        Prompt: \"{prompt_preview}\"")
                    print(f"        File: {prompt_info[name]['file']}")
                    print(f"        Dim: {prompt_info[name]['vector_dim']}")
        else:
            print(f"\n✗ Could not parse view mapping!")
        
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON: {e}")

if __name__ == '__main__':
    datasets = ['Amazon_Beauty', 'Amazon_Toys_and_Games']
    
    for dataset in datasets:
        check_views_json(dataset)
    
    print(f"\n{'='*80}\n")

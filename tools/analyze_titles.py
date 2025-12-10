#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Title Data Analyzer for Multi-View Prompt Design

This tool analyzes item titles from different datasets to identify:
1. Text characteristics (length, vocabulary, patterns)
2. Information density (entities, attributes, modifiers)
3. Semantic dimensions (what aspects are commonly mentioned)
4. Cross-dataset differences

Output: JSON report + recommendations for universal prompt design

Usage:
    python tools/analyze_titles.py \
        --datasets dataset/Amazon_Beauty dataset/Amazon_Toys_and_Games \
        --names Beauty Toys \
        --output title_analysis_report.json
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm


class TitleAnalyzer:
    """Analyze item titles to inform prompt design"""
    
    def __init__(self, titles: List[str], dataset_name: str):
        self.titles = [str(t).strip() for t in titles if t and str(t).strip()]
        self.dataset_name = dataset_name
        self.stats = {}
        
    def analyze(self) -> Dict[str, Any]:
        """Run all analyses"""
        print(f"\n[{self.dataset_name}] Analyzing {len(self.titles)} titles...")
        
        self.stats = {
            "dataset": self.dataset_name,
            "total_titles": len(self.titles),
            "basic_stats": self._basic_statistics(),
            "vocabulary": self._vocabulary_analysis(),
            "patterns": self._pattern_analysis(),
            "semantic_signals": self._semantic_signal_analysis(),
            "information_density": self._information_density(),
        }
        
        return self.stats
    
    def _basic_statistics(self) -> Dict[str, Any]:
        """Basic text statistics"""
        lengths = [len(t) for t in self.titles]
        word_counts = [len(t.split()) for t in self.titles]
        
        return {
            "char_length": {
                "mean": float(np.mean(lengths)),
                "std": float(np.std(lengths)),
                "min": int(np.min(lengths)),
                "max": int(np.max(lengths)),
                "median": float(np.median(lengths)),
                "q25": float(np.percentile(lengths, 25)),
                "q75": float(np.percentile(lengths, 75)),
            },
            "word_count": {
                "mean": float(np.mean(word_counts)),
                "std": float(np.std(word_counts)),
                "min": int(np.min(word_counts)),
                "max": int(np.max(word_counts)),
                "median": float(np.median(word_counts)),
                "q25": float(np.percentile(word_counts, 25)),
                "q75": float(np.percentile(word_counts, 75)),
            },
            "empty_ratio": sum(1 for t in self.titles if len(t) < 3) / max(len(self.titles), 1),
        }
    
    def _vocabulary_analysis(self) -> Dict[str, Any]:
        """Vocabulary richness and diversity"""
        all_words = []
        for title in self.titles:
            words = re.findall(r'\b[a-zA-Z]+\b', title.lower())
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        unique_words = len(word_freq)
        total_words = len(all_words)
        
        # Type-Token Ratio (TTR) - diversity metric
        ttr = unique_words / max(total_words, 1)
        
        # Top words
        top_50_words = word_freq.most_common(50)
        top_50_coverage = sum(count for _, count in top_50_words) / max(total_words, 1)
        
        return {
            "unique_words": unique_words,
            "total_words": total_words,
            "type_token_ratio": ttr,
            "top_50_words": [{"word": w, "count": c, "freq": c/total_words} for w, c in top_50_words[:20]],
            "top_50_coverage": top_50_coverage,
            "avg_word_length": np.mean([len(w) for w in all_words]) if all_words else 0,
        }
    
    def _pattern_analysis(self) -> Dict[str, Any]:
        """Common patterns and structures"""
        patterns = {
            "has_numbers": 0,
            "has_units": 0,  # oz, ml, cm, inch, etc.
            "has_parentheses": 0,
            "has_dash": 0,
            "has_comma": 0,
            "has_slash": 0,
            "has_ampersand": 0,
            "all_uppercase_words": 0,
            "has_brand_like": 0,  # Starts with capitalized word
        }
        
        unit_pattern = r'\b\d+\s*(oz|ml|lb|kg|cm|inch|mm|ft|pack|count|piece|set)\b'
        
        for title in self.titles:
            if re.search(r'\d', title):
                patterns["has_numbers"] += 1
            if re.search(unit_pattern, title.lower()):
                patterns["has_units"] += 1
            if '(' in title or ')' in title:
                patterns["has_parentheses"] += 1
            if '-' in title:
                patterns["has_dash"] += 1
            if ',' in title:
                patterns["has_comma"] += 1
            if '/' in title:
                patterns["has_slash"] += 1
            if '&' in title:
                patterns["has_ampersand"] += 1
            
            words = title.split()
            if words and any(w.isupper() and len(w) > 1 for w in words):
                patterns["all_uppercase_words"] += 1
            if words and words[0] and words[0][0].isupper():
                patterns["has_brand_like"] += 1
        
        # Convert counts to ratios
        n = max(len(self.titles), 1)
        return {k: v / n for k, v in patterns.items()}
    
    def _semantic_signal_analysis(self) -> Dict[str, Any]:
        """Analyze what semantic dimensions are present in titles"""
        
        # Keywords indicating different semantic aspects
        semantic_keywords = {
            "material": [
                "plastic", "metal", "wood", "glass", "ceramic", "leather", 
                "cotton", "silk", "nylon", "rubber", "steel", "aluminum",
                "fabric", "paper", "cardboard", "foam", "vinyl"
            ],
            "color": [
                "red", "blue", "green", "yellow", "black", "white", "pink",
                "purple", "orange", "brown", "gray", "silver", "gold", "clear"
            ],
            "size": [
                "large", "small", "medium", "mini", "big", "tiny", "compact",
                "xl", "xs", "xxl", "travel", "size", "full", "queen", "king"
            ],
            "age_group": [
                "baby", "kids", "children", "toddler", "infant", "teen",
                "adult", "boy", "girl", "men", "women", "unisex"
            ],
            "function": [
                "moisturizing", "cleansing", "anti", "repair", "protect",
                "waterproof", "educational", "interactive", "play", "learn"
            ],
            "quality": [
                "premium", "professional", "natural", "organic", "luxury",
                "deluxe", "classic", "vintage", "modern", "new", "improved"
            ],
            "quantity": [
                "pack", "set", "bundle", "collection", "kit", "piece",
                "count", "oz", "ml", "lb"
            ],
            "brand_indicators": [
                "by", "from", "original", "authentic", "official", "licensed"
            ],
            "usage_context": [
                "outdoor", "indoor", "travel", "home", "office", "school",
                "party", "gift", "birthday", "wedding", "christmas"
            ],
        }
        
        coverage = {}
        matched_examples = {}
        
        for dimension, keywords in semantic_keywords.items():
            matched_titles = []
            for title in self.titles:
                title_lower = title.lower()
                if any(kw in title_lower for kw in keywords):
                    matched_titles.append(title)
                    if len(matched_examples.get(dimension, [])) < 3:
                        matched_examples.setdefault(dimension, []).append(title)
            
            coverage[dimension] = {
                "ratio": len(matched_titles) / max(len(self.titles), 1),
                "count": len(matched_titles),
                "examples": matched_examples.get(dimension, [])[:3],
            }
        
        return coverage
    
    def _information_density(self) -> Dict[str, Any]:
        """Measure how much information is packed in titles"""
        
        def count_entities(title: str) -> int:
            """Count distinct information units (brands, numbers, attributes)"""
            count = 0
            # Numbers (size, quantity, etc.)
            count += len(re.findall(r'\d+', title))
            # Capitalized words (likely brands/proper nouns)
            count += len(re.findall(r'\b[A-Z][a-z]+', title))
            # Parenthetical info
            count += len(re.findall(r'\([^)]+\)', title))
            return count
        
        entity_counts = [count_entities(t) for t in self.titles]
        word_counts = [len(t.split()) for t in self.titles]
        
        # Information density = entities per word
        densities = [
            e / max(w, 1) for e, w in zip(entity_counts, word_counts)
        ]
        
        return {
            "avg_entities_per_title": float(np.mean(entity_counts)),
            "avg_density": float(np.mean(densities)),
            "std_density": float(np.std(densities)),
            "high_density_ratio": sum(1 for d in densities if d > 0.5) / max(len(densities), 1),
        }


class MultiDatasetAnalyzer:
    """Compare multiple datasets and generate prompt recommendations"""
    
    def __init__(self, dataset_paths: List[str], dataset_names: List[str]):
        self.dataset_paths = dataset_paths
        self.dataset_names = dataset_names
        self.analyzers = []
        self.results = {}
        
    def load_titles(self, dataset_path: str) -> List[str]:
        """Load titles from dataset mapping CSV"""
        mapping_file = os.path.join(dataset_path, "item_index_mapping.csv")
        
        if not os.path.exists(mapping_file):
            raise FileNotFoundError(f"Missing: {mapping_file}")
        
        df = pd.read_csv(mapping_file)
        
        if "title" in df.columns:
            titles = df["title"].tolist()
        elif "item_token" in df.columns:
            titles = df["item_token"].tolist()
        else:
            raise ValueError(f"No 'title' or 'item_token' column in {mapping_file}")
        
        return titles
    
    def analyze_all(self) -> Dict[str, Any]:
        """Analyze all datasets"""
        for path, name in zip(self.dataset_paths, self.dataset_names):
            titles = self.load_titles(path)
            analyzer = TitleAnalyzer(titles, name)
            stats = analyzer.analyze()
            self.results[name] = stats
            self.analyzers.append(analyzer)
        
        return self.results
    
    def compare_datasets(self) -> Dict[str, Any]:
        """Cross-dataset comparison"""
        if len(self.results) < 2:
            return {}
        
        comparison = {
            "datasets_compared": list(self.results.keys()),
            "differences": {},
        }
        
        # Compare key metrics
        names = list(self.results.keys())
        
        for metric_path in [
            ("basic_stats", "word_count", "mean"),
            ("basic_stats", "char_length", "mean"),
            ("vocabulary", "type_token_ratio"),
            ("information_density", "avg_density"),
        ]:
            values = []
            for name in names:
                val = self.results[name]
                for key in metric_path:
                    val = val.get(key, {})
                    if not isinstance(val, dict):
                        break
                values.append(val if not isinstance(val, dict) else 0)
            
            metric_name = "_".join(metric_path)
            comparison["differences"][metric_name] = {
                name: float(val) for name, val in zip(names, values)
            }
        
        # Semantic coverage comparison
        semantic_comparison = {}
        if len(names) >= 2:
            for dimension in self.results[names[0]].get("semantic_signals", {}).keys():
                semantic_comparison[dimension] = {}
                for name in names:
                    ratio = self.results[name]["semantic_signals"][dimension]["ratio"]
                    semantic_comparison[dimension][name] = float(ratio)
        
        comparison["semantic_coverage_comparison"] = semantic_comparison
        
        return comparison
    
    def generate_prompt_recommendations(self) -> Dict[str, Any]:
        """Generate prompt design recommendations based on analysis"""
        
        recommendations = {
            "summary": "",
            "universal_dimensions": [],
            "dataset_specific_notes": {},
            "suggested_prompts": [],
        }
        
        if not self.results:
            return recommendations
        
        # Analyze semantic coverage across all datasets
        all_dimensions = set()
        for result in self.results.values():
            all_dimensions.update(result.get("semantic_signals", {}).keys())
        
        # Find universally relevant dimensions (>20% coverage in all datasets)
        universal_dims = []
        for dim in all_dimensions:
            min_coverage = min(
                self.results[name]["semantic_signals"].get(dim, {}).get("ratio", 0)
                for name in self.results.keys()
            )
            if min_coverage > 0.2:
                universal_dims.append((dim, min_coverage))
        
        universal_dims.sort(key=lambda x: x[1], reverse=True)
        recommendations["universal_dimensions"] = [
            {"dimension": dim, "min_coverage": float(cov)}
            for dim, cov in universal_dims[:5]
        ]
        
        # Dataset-specific characteristics
        for name, result in self.results.items():
            notes = []
            
            # Word count
            avg_words = result["basic_stats"]["word_count"]["mean"]
            if avg_words < 5:
                notes.append("Very short titles - focus on explicit attribute extraction")
            elif avg_words > 10:
                notes.append("Long titles - can extract multiple attributes")
            
            # Information density
            density = result["information_density"]["avg_density"]
            if density > 0.6:
                notes.append("High information density - rich in brands/numbers/attributes")
            elif density < 0.3:
                notes.append("Low information density - may need more inferential prompts")
            
            # Top semantic signals
            top_signals = sorted(
                result["semantic_signals"].items(),
                key=lambda x: x[1]["ratio"],
                reverse=True
            )[:3]
            notes.append(
                f"Strongest signals: {', '.join(s[0] for s in top_signals)}"
            )
            
            recommendations["dataset_specific_notes"][name] = notes
        
        # Suggested prompt templates
        recommendations["suggested_prompts"] = self._generate_prompt_suggestions()
        
        # Summary
        n_datasets = len(self.results)
        avg_word_count = np.mean([
            r["basic_stats"]["word_count"]["mean"] for r in self.results.values()
        ])
        
        recommendations["summary"] = (
            f"Analyzed {n_datasets} datasets. "
            f"Average title length: {avg_word_count:.1f} words. "
            f"Found {len(universal_dims)} universal semantic dimensions. "
            f"Recommend prompts that balance explicit extraction with inference."
        )
        
        return recommendations
    
    def _generate_prompt_suggestions(self) -> List[Dict[str, str]]:
        """Generate concrete prompt suggestions"""
        
        # Analyze what works across datasets
        suggestions = []
        
        # Always useful: functional description (what it does)
        suggestions.append({
            "view": "function",
            "prompt": "What are the main functions and features of [TITLE] {text}?",
            "rationale": "Functional attributes are present across product domains",
        })
        
        # User targeting (who it's for)
        suggestions.append({
            "view": "audience",
            "prompt": "Who is the ideal user or target audience for [TITLE] {text}?",
            "rationale": "Age groups, gender, user types are commonly inferable",
        })
        
        # Check if physical attributes are common
        avg_material_coverage = np.mean([
            r["semantic_signals"].get("material", {}).get("ratio", 0)
            for r in self.results.values()
        ])
        
        if avg_material_coverage > 0.15:
            suggestions.append({
                "view": "physical",
                "prompt": "Describe the physical attributes of [TITLE] {text}: materials, size, color, and design.",
                "rationale": f"Physical attributes mentioned in {avg_material_coverage*100:.1f}% of titles",
            })
        else:
            suggestions.append({
                "view": "context",
                "prompt": "When and where would someone typically use [TITLE] {text}?",
                "rationale": "Usage context provides orthogonal information to function",
            })
        
        # Quality/value proposition
        suggestions.append({
            "view": "value",
            "prompt": "What makes [TITLE] {text} valuable or appealing to buyers?",
            "rationale": "Captures implicit quality signals and emotional appeal",
        })
        
        return suggestions


def main():
    parser = argparse.ArgumentParser(
        description="Analyze item titles to inform multi-view prompt design"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Paths to dataset directories (must contain item_index_mapping.csv)",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="Short names for datasets (e.g., Beauty Toys)",
    )
    parser.add_argument(
        "--output",
        default="title_analysis_report.json",
        help="Output JSON file path",
    )
    
    args = parser.parse_args()
    
    if len(args.datasets) != len(args.names):
        raise ValueError("Number of --datasets must match --names")
    
    # Run analysis
    analyzer = MultiDatasetAnalyzer(args.datasets, args.names)
    results = analyzer.analyze_all()
    comparison = analyzer.compare_datasets()
    recommendations = analyzer.generate_prompt_recommendations()
    
    # Compile report
    report = {
        "individual_analyses": results,
        "cross_dataset_comparison": comparison,
        "recommendations": recommendations,
    }
    
    # Save report
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Report saved to: {os.path.abspath(args.output)}")
    print(f"\n{recommendations['summary']}")
    print(f"\nUniversal dimensions found:")
    for dim in recommendations["universal_dimensions"]:
        print(f"  - {dim['dimension']}: {dim['min_coverage']*100:.1f}% min coverage")
    
    print(f"\nSuggested prompts:")
    for i, prompt in enumerate(recommendations["suggested_prompts"], 1):
        print(f"  {i}. [{prompt['view']}] {prompt['prompt']}")
        print(f"     → {prompt['rationale']}")
    
    print(f"\nDataset-specific notes:")
    for name, notes in recommendations["dataset_specific_notes"].items():
        print(f"  {name}:")
        for note in notes:
            print(f"    - {note}")


if __name__ == "__main__":
    main()

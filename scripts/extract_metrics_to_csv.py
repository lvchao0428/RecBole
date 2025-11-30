#!/usr/bin/env python3
"""
Extract metric names and values from a log line or text blob containing a (Ordered)Dict
like either:
- "INFO test result: OrderedDict({'recall@5': np.float64(0.0347), ...})"
- "INFO test result: OrderedDict([('recall@5', 0.0347), ('recall@10', 0.05), ...])"
and output two CSV lines: a header row of names and a data row of values.

Usage examples:
  # Read from stdin and print to stdout
  echo "07 Nov 20:27 INFO test result: OrderedDict({'recall@5': np.float64(0.0347)})" | \\
    python3 scripts/extract_metrics_to_csv.py

  # Parse modern Run Summary JSON and focus on test metrics
  python3 scripts/extract_metrics_to_csv.py --dict-path results.test_result --from-string "$(pbpaste)"

  # Read from file and write to CSV
  python3 scripts/extract_metrics_to_csv.py -f path/to/log.txt -o metrics.csv

  # Pass the line directly
  python3 scripts/extract_metrics_to_csv.py --from-string "07 Nov ... OrderedDict({'recall@5': np.float64(0.0347)})"
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import re
import sys
from typing import Dict, Iterable, List, Tuple, Union

METRIC_KEY_PREFIXES = (
    "recall@",
    "mrr@",
    "ndcg@",
    "hit@",
    "precision@",
    "map@",
    "auc@",
)

DEFAULT_DICT_PATHS = [
    "results.phase_b_result.test_result",
    "results.phase_b_result.best_valid_result",
    "results.test_result",
    "results.best_valid_result",
]


def read_all_input(args: argparse.Namespace) -> str:
    if args.from_string is not None:
        return args.from_string
    if args.file is not None:
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read()
    # stdin
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("No input provided. Use --from-string, --file, or pipe data via stdin.")


def extract_braced_substring(text: str) -> str:
    """
    Try to extract the substring representing the dict literal by taking the
    first '{' to the last '}'.
    Falls back to original text if braces not found.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def sanitize_wrappers(text: str) -> str:
    """
    Remove wrappers like np.float64(0.123), numpy.float32(0.1), Decimal(0.2), float(0.3).
    Convert them to the inner numeric literal so that ast.literal_eval can parse it.
    """
    patterns = [
        r"np\.float(?:16|32|64)?\(([^)]+)\)",
        r"numpy\.float(?:16|32|64)?\(([^)]+)\)",
        r"Decimal\(([^)]+)\)",
        r"float\(([^)]+)\)",
    ]
    sanitized = text
    for pat in patterns:
        sanitized = re.sub(pat, r"\1", sanitized)
    return sanitized


def looks_like_metric_mapping(candidate) -> bool:
    if not isinstance(candidate, dict):
        return False
    matches = 0
    for key in candidate.keys():
        key_lower = str(key).lower()
        if any(key_lower.startswith(prefix) for prefix in METRIC_KEY_PREFIXES):
            matches += 1
        if matches >= 2:
            return True
    return False


def get_by_path(mapping: dict, path: str):
    if not isinstance(mapping, dict):
        return None
    current = mapping
    for raw_part in path.split("."):
        part = raw_part.strip()
        if not part:
            continue
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def discover_metric_mapping(obj):
    stack = [obj]
    while stack:
        current = stack.pop()
        if looks_like_metric_mapping(current):
            return current
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(item for item in current if not isinstance(item, (str, bytes)))
    return None


def select_metric_mapping(mapping: dict, user_paths: List[str] | None = None):
    search_paths: List[str] = []
    if user_paths:
        search_paths.extend(user_paths)
    search_paths.extend([p for p in DEFAULT_DICT_PATHS if p not in (user_paths or [])])

    for path in search_paths:
        candidate = get_by_path(mapping, path)
        if isinstance(candidate, dict):
            if user_paths and path in user_paths:
                return candidate
            if looks_like_metric_mapping(candidate):
                return candidate
    auto = discover_metric_mapping(mapping)
    if isinstance(auto, dict):
        return auto
    return mapping


def try_literal_eval_dict(text: str) -> Union[Dict[str, Union[int, float, str]], None]:
    try:
        value = ast.literal_eval(text)
    except Exception:
        try:
            value = json.loads(text)
    except Exception:
        return None
    if isinstance(value, dict):
        return value
    return None


def regex_parse_kv(text: str) -> Tuple[List[str], List[str]]:
    """
    Fallback parser: extract 'key': value pairs using regex.
    Values are kept as strings and later converted to numbers when possible.
    """
    # This matches keys in single or double quotes, followed by colon, then captures
    # a simple value up to the next comma or closing brace.
    pattern = re.compile(r"""(['"])(.*?)\1\s*:\s*([^,}]+)""")
    keys: List[str] = []
    vals: List[str] = []
    for m in pattern.finditer(text):
        keys.append(m.group(2))
        vals.append(m.group(3).strip())
    if not keys:
        raise ValueError("Could not parse any key/value pairs.")
    return keys, vals


def try_parse_ordereddict_tuple_list(text: str) -> Union[Tuple[List[str], List[str]], None]:
    """
    Handle logs in the form:
      "OrderedDict([('recall@5', 0.0333), ('recall@10', 0.0458), ...])"
    Returns keys and raw string values if detected, else None.
    """
    # Find the content inside OrderedDict([...])
    m = re.search(r"OrderedDict\(\s*\[([\s\S]*?)\]\s*\)", text)
    if not m:
        return None
    content = m.group(1)
    # Now capture each tuple ('key', value)
    pair_re = re.compile(r"""\(\s*(['"])(.*?)\1\s*,\s*([^)]+?)\s*\)""")
    keys: List[str] = []
    vals: List[str] = []
    for pm in pair_re.finditer(content):
        keys.append(pm.group(2))
        vals.append(pm.group(3).strip())
    if not keys:
        return None
    return keys, vals


def coerce_to_number(value: Union[int, float, str]) -> Union[int, float, str]:
    if isinstance(value, (int, float)):
        return value
    s = str(value).strip()
    # Remove possible quotes
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1]
    try:
        # Try int first to keep clean integers when applicable
        if re.fullmatch(r"[+-]?\d+", s):
            return int(s)
        return float(s)
    except Exception:
        return s


def format_value_for_csv(value: Union[int, float, str]) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # Avoid excessive precision while keeping significant digits
        return f"{value:.12g}"
    return str(value)


def to_csv_rows(mapping: Dict[str, Union[int, float, str]]) -> Tuple[List[str], List[str]]:
    keys: List[str] = list(mapping.keys())
    vals: List[str] = [format_value_for_csv(coerce_to_number(mapping[k])) for k in keys]
    return keys, vals


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract metrics dict to two-line CSV (header + values).")
    parser.add_argument("-f", "--file", dest="file", help="Input file containing the log/text.")
    parser.add_argument("--from-string", dest="from_string", help="Input text provided directly.")
    parser.add_argument("-o", "--output", dest="output", help="Write CSV to this file. Prints to stdout if omitted.")
    parser.add_argument("-d", "--delimiter", default=",", help="CSV delimiter (default: ',').")
    parser.add_argument("--encoding", default="utf-8", help="Output file encoding (default: utf-8).")
    parser.add_argument(
        "--dict-path",
        dest="dict_paths",
        action="append",
        help="Dot notation path to nested metrics dict (e.g., 'results.test_result'). "
        "You can specify multiple times; first match wins before auto-detection.",
    )
    args = parser.parse_args(argv)

    raw_text = read_all_input(args)
    dict_like = extract_braced_substring(raw_text)
    sanitized = sanitize_wrappers(dict_like)

    mapping = try_literal_eval_dict(sanitized)
    if mapping is None:
        # Try OrderedDict with list-of-tuples format
        od_result = try_parse_ordereddict_tuple_list(sanitized)
        if od_result is not None:
            keys, raw_vals = od_result
        else:
            # Fallback: regex parse of dict-like "'key': value" pairs
            keys, raw_vals = regex_parse_kv(sanitized)
        values = [format_value_for_csv(coerce_to_number(v)) for v in raw_vals]
    else:
        metric_mapping = select_metric_mapping(mapping, args.dict_paths)
        keys, values = to_csv_rows(metric_mapping)

    if args.output:
        with open(args.output, "w", newline="", encoding=args.encoding) as f:
            writer = csv.writer(f, delimiter=args.delimiter)
            writer.writerow(keys)
            writer.writerow(values)
    else:
        # Print to stdout
        stdout = io.StringIO()
        writer = csv.writer(stdout, delimiter=args.delimiter, lineterminator="\n")
        writer.writerow(keys)
        writer.writerow(values)
        sys.stdout.write(stdout.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



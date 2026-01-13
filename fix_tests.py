#!/usr/bin/env python3
"""Script to fix all add_junction calls in test files and example scripts."""

import re
from pathlib import Path


def fix_add_junction_call(match):
    """Convert old add_junction syntax to new syntax."""
    # Extract the parameters
    full_match = match.group(0)
    var_name = match.group(1)  # Capture the variable name (skel or skeleton)

    # Parse id, xyz array, and radius
    id_match = re.search(r"id=(\d+)", full_match)
    xyz_match = re.search(r"xyz=np\.array\(\[([^\]]+)\]\)", full_match)
    radius_match = re.search(r"radius=([\d.]+)", full_match)

    if not (id_match and xyz_match and radius_match):
        return full_match  # Can't parse, return unchanged

    node_id = id_match.group(1)
    xyz_values = [x.strip() for x in xyz_match.group(1).split(",")]
    radius = radius_match.group(1)

    if len(xyz_values) != 3:
        return full_match  # Invalid xyz, return unchanged

    x, y, z = xyz_values

    # Build new call with the original variable name
    return (
        f"{var_name}.add_junction(node_id={node_id}, x={x}, y={y}, z={z}, r={radius})"
    )


def fix_file(filepath):
    """Fix add_junction calls in a single file."""
    content = filepath.read_text(encoding="utf-8")
    original = content

    # Pattern to match old-style add_junction calls (captures variable name)
    pattern = r"(skel(?:eton)?|skeleton)\.add_junction\(id=\d+,\s*xyz=np\.array\(\[[^\]]+\]\),\s*radius=[\d.]+\)"

    content = re.sub(pattern, fix_add_junction_call, content)

    if content != original:
        filepath.write_text(content, encoding="utf-8")
        print(f"Fixed: {filepath}")
        return True
    return False


def main():
    """Fix all test and example files."""
    root = Path(__file__).parent

    # Fix test files
    test_dir = root / "tests"
    example_dir = root / "examples"

    fixed_count = 0

    for directory in [test_dir, example_dir]:
        if directory.exists():
            for py_file in directory.glob("*.py"):
                if fix_file(py_file):
                    fixed_count += 1

    print(f"\nTotal files fixed: {fixed_count}")


if __name__ == "__main__":
    main()

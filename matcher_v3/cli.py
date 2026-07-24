"""
Command-line interface for fabric pattern matching.

Usage:
    # Build index
    python -m matcher_v3.cli build 素材/fabric/

    # Single query
    python -m matcher_v3.cli search 素材/photo/1.png

    # Evaluate all
    python -m matcher_v3.cli eval

    # Evaluate with custom paths
    python -m matcher_v3.cli eval --photo-dir 素材/photo/ --fabric-dir 素材/fabric/
"""
import argparse
import os
import sys
import time

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')


def cmd_build(args):
    """Build (or rebuild) the fabric library index."""
    from .indexer import FabricIndex

    index = FabricIndex()
    t0 = time.time()
    index.build(args.fabric_dir)
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s.")


def cmd_search(args):
    """Search for matching fabrics for a single query photo."""
    from .indexer import FabricIndex
    from .matcher import FabricMatcher

    # Load index
    index = FabricIndex()
    if not index.load(CACHE_DIR):
        print("Error: No index found. Run 'build' first.", file=sys.stderr)
        sys.exit(1)

    matcher = FabricMatcher(index)
    results = matcher.match(args.photo_path, top_n=args.top_n, verbose=True)

    # Return top result name as exit hint
    if results:
        print(f"\n  Best match: {results[0]['name']} (score={results[0]['final_score']:.4f})")


def cmd_eval(args):
    """Evaluate matching accuracy on all photo/fabric pairs."""
    from .indexer import FabricIndex
    from .matcher import FabricMatcher

    index = FabricIndex()
    index.build(args.fabric_dir)

    matcher = FabricMatcher(index)

    photo_dir = args.photo_dir
    fabric_dir = args.fabric_dir

    print(f"\n{'=' * 80}")
    print(f"  Evaluation: {photo_dir} -> {fabric_dir}")
    print(f"{'=' * 80}")

    stats = matcher.eval_all(photo_dir, fabric_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Fabric Pattern Matcher v3 — Three-stage cascade matching'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # build
    p_build = subparsers.add_parser('build', help='Build fabric library index')
    p_build.add_argument('fabric_dir', help='Path to fabric pattern library directory')

    # search
    p_search = subparsers.add_parser('search', help='Search for matching fabrics')
    p_search.add_argument('photo_path', help='Path to query photo')
    p_search.add_argument('--top-n', type=int, default=5, help='Number of results to show')

    # eval
    p_eval = subparsers.add_parser('eval', help='Evaluate accuracy on all pairs')
    p_eval.add_argument('--photo-dir', default='dir/photo/', help='Photo directory')
    p_eval.add_argument('--fabric-dir', default='dir/fabric/', help='Fabric directory')
    p_eval.add_argument('--verbose', '-v', action='store_true', help='Show per-photo details')

    args = parser.parse_args()

    if args.command == 'build':
        cmd_build(args)
    elif args.command == 'search':
        cmd_search(args)
    elif args.command == 'eval':
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

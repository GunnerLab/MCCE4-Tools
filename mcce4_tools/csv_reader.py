#!/usr/bin/env python3
import argparse
import sys
import pandas as pd

def parse_arguments():
    """Configures the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="A CLI tool to read and preview CSV files using Pandas."
    )

    # Required argument: The path to the file
    parser.add_argument(
        "filepath", type=str, help="Path to the CSV file you want to read"
    )

    # Optional argument: Change the delimiter (Default is comma)
    parser.add_argument(
        "-s",
        "--sep",
        type=str,
        default=",",
        help="Column delimiter/separator (default: ',')",
    )

    # Optional argument: Choose number of rows to print
    # Default is 5. User can pass -1 to print all rows.
    parser.add_argument(
        "-n",
        "--rows",
        type=int,
        default=5,
        help="Number of rows to display (use -1 for all rows, default: 5)",
    )

    # Optional flag: Show full technical summary of the data
    parser.add_argument(
        "-i",
        "--info",
        action="store_true",
        help="Show data types, column names, and missing value counts",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        # Set Pandas options to prevent truncation in the terminal
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        # Load the CSV
        df = pd.read_csv(args.filepath, sep=args.sep)

        # Print layout formatting
        print(f"\n=== Successfully loaded: {args.filepath} ===")
        print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

        # Action 1: Print dataframe technical details if requested
        if args.info:
            print("--- Dataframe Summary ---")
            df.info()
            print("\n")

        # Action 2: Print data preview
        print(f"--- Displaying {'All' if args.rows == -1 else args.rows} Rows ---")
        
        if args.rows == -1:
            # Print everything
            print(df.to_string())
        else:
            # Print specific number of rows
            print(df.head(args.rows).to_string())
            
        print("=" * (25 + len(args.filepath)))

    except FileNotFoundError:
        print(
            f"Error: The file '{args.filepath}' was not found. Check the path.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

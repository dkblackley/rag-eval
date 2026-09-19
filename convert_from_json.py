import sys
import json


def convert_json_to_tsv(json_path, tsv_path):
    """
    Reads a JSON file mapping QIDs to lists of result IDs,
    and writes them to a TSV format, preserving the list order.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        query_results = json.load(f)

    with open(tsv_path, 'w', encoding='utf-8') as f:
        for qid, results in query_results.items():
            for result_id in results:
                # Adding the '1.0' to perfectly match the original TSV format
                f.write(f"{qid}\t{result_id}\t1.0\n")


if __name__ == "__main__":
    # Ensure the user provided exactly two arguments
    if len(sys.argv) != 3:
        print("Usage: python json_to_tsv.py <input.json> <output.tsv>")
        sys.exit(1)

    input_json = sys.argv[1]
    output_tsv = sys.argv[2]

    convert_json_to_tsv(input_json, output_tsv)
    print(f"Conversion complete. Saved TSV to: {output_tsv}")
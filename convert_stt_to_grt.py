import json
import argparse

def read_json_file(file_path):
    """Read JSON data from a file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {e}")
        raise

def transform_data(input_data, modality_data):
    """Transform input data into the desired format based on modality structure."""
    # Initialize output structure
    output_data = {
        "observation.state":{"mean": [], "std": [], "min": [], "max": [], "q01": [], "q99": []},
        "action": {"mean": [], "std": [], "min": [], "max": [], "q01": [], "q99": []}
    }

    # Define the order of modalities and their corresponding input keys
    state_modalities = [
        ("qpos_left_arm", "qpos", 0, 7),   # qpos_0 to qpos_6
        ("qpos_right_arm", "qpos", 7, 14), # qpos_7 to qpos_13
        ("qvel_left_arm", "qvel", 0, 7),   # qvel_0 to qvel_6
        ("qvel_right_arm", "qvel", 7, 14), # qvel_7 to qvel_13
        ("ee_left_arm", "ee", 0, 7),     # ee_0 to ee_6
        ("ee_right_arm", "ee", 7, 14),    # ee_7 to ee_13
        ("torque_left_arm", "torque", 0, 7), # torque_0 to torque_6
        ("torque_right_arm", "torque", 7, 14) # torque_7 to torq_ue13
    ]
    action_modalities = [
        ("action_left_arm", "action", 0, 7), # action_0 to action_6
        ("action_right_arm", "action", 7, 14) # action_7 to actio_n13
    ]

    # Helper function to collect values for a modality
    def collect_values(modality_name, prefix, start_idx, end_idx, target, input_stats):
        """Collect statistical values for a specific modality range."""
        for idx in range(start_idx, end_idx):
            key = f"{prefix}_{idx}"
            if key not in input_data:
                raise ValueError(f"Missing key {key} in input data")
            for stat in ["mean", "std", "min", "max", "q01", "q99"]:
                # Extract single value from list
                target[stat].append(input_data[key][stat][0])

    # Process state modalities
    for modality_name, prefix, start_idx, end_idx in state_modalities:
        collect_values(modality_name, prefix, start_idx, end_idx, output_data["observation.state"], input_data)

    # Process action modalities
    for modality_name, prefix, start_idx, end_idx in action_modalities:
        collect_values(modality_name, prefix, start_idx, end_idx, output_data["action"], input_data)

    return output_data

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert JSON stats to desired format")
    parser.add_argument("--input-stats", required=True, help="Path to input statistical JSON file")
    parser.add_argument("--input-modality", required=True, help="Path to modality JSON file")
    parser.add_argument("--output", default="stats_transformed.json", help="Path to output JSON file")

    args = parser.parse_args()

    # Read input files
    input_data = read_json_file(args.input_stats)
    modality_data = read_json_file(args.input_modality)

    # Transform data
    output_data = transform_data(input_data, modality_data)

    # Write output to file
    try:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Transformation complete. Output saved to '{args.output}'")
    except Exception as e:
        print(f"Error writing to output file {args.output}: {e}")
        raise

if __name__ == "__main__":
    main()
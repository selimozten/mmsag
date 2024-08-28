import os

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def collect_project_data(root_dir, output_file):
    relevant_extensions = ['.py', '.yml', '.yaml', '.txt', '.md', '.sh']
    project_data = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(ext) for ext in relevant_extensions):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, root_dir)
                try:
                    content = read_file(file_path)
                    project_data.append(f"File: {relative_path}\n\n{content}\n\n{'='*80}\n")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write("\n".join(project_data))

def main():
    root_dir = '.'  # Assumes the script is run from the project root directory
    output_file = 'project_data.txt'
    collect_project_data(root_dir, output_file)
    print(f"Project data collected and saved to {output_file}")

if __name__ == "__main__":
    main()
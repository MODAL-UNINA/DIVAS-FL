# %%
import os


def extract_id_num(path):
    # Path example: ../data/train/id00001/audio/00001.pt
    parts = path.split(os.sep)
    id_part = parts[-3]
    filename = parts[-1]
    num_part = os.path.splitext(filename)[0]
    return (id_part, num_part)


def filter_common_files(run_txt_dir, split):
    paths_audio = []
    paths_f0 = []
    paths_f0_norm = []

    with open(os.path.join(run_txt_dir, f"{split}_wav.txt"), 'r') as f:
        paths_audio = [line.strip() for line in f.readlines()]
    with open(os.path.join(run_txt_dir, f"{split}_f0.txt"), 'r') as f:
        paths_f0 = [line.strip() for line in f.readlines()]
    with open(os.path.join(run_txt_dir, f"{split}_f0_norm.txt"), 'r') as f:
        paths_f0_norm = [line.strip() for line in f.readlines()]

    # Extract id and num from paths
    audio_pairs = set(extract_id_num(p) for p in paths_audio)
    f0_pairs = set(extract_id_num(p) for p in paths_f0)
    f0_norm_pairs = set(extract_id_num(p) for p in paths_f0_norm)

    common_pairs = audio_pairs & f0_pairs & f0_norm_pairs

    print(f"{split}: found {len(audio_pairs)} audio, {len(f0_pairs)} f0, {len(f0_norm_pairs)} f0_norm")
    print(f"{split}: {len(common_pairs)} common files found")

    def filter_list(paths):
        return [p for p in paths if extract_id_num(p) in common_pairs]

    filtered_audio = filter_list(paths_audio)
    filtered_f0 = filter_list(paths_f0)
    filtered_f0_norm = filter_list(paths_f0_norm)

    with open(os.path.join(run_txt_dir, f"{split}_wav.txt"), 'w') as f:
        f.write('\n'.join(filtered_audio) + '\n')
    with open(os.path.join(run_txt_dir, f"{split}_f0.txt"), 'w') as f:
        f.write('\n'.join(filtered_f0) + '\n')
    with open(os.path.join(run_txt_dir, f"{split}_f0_norm.txt"), 'w') as f:
        f.write('\n'.join(filtered_f0_norm) + '\n')


def generate_filelist(data_root, split='train', subfolder='audio', output_file='filelist.txt'):
    split_path = os.path.join(data_root, split)
    file_paths = []

    for speaker_id in os.listdir(split_path):
        speaker_path = os.path.join(split_path, speaker_id)
        if not os.path.isdir(speaker_path):
            continue

        target_folder = os.path.join(speaker_path, subfolder)
        if not os.path.isdir(target_folder):
            print(f"[Warning] No '{subfolder}' folder for {speaker_id} in {split}")
            continue

        for root, _, files in os.walk(target_folder):
            for file in files:
                if file.endswith('.wav') and subfolder == 'audio':
                    full_path = os.path.join(root, file)
                elif subfolder in ['f0', 'f0_norm']:
                    full_path = os.path.join(root, file)
                else:
                    continue

                rel_path = os.path.relpath(full_path, start=os.getcwd())
                file_paths.append(rel_path)

    with open(output_file, 'w') as f:
        for path in sorted(file_paths):
            f.write('../' + path + '\n')

    print(f"File {output_file} created with {len(file_paths)} paths relative to {os.getcwd()}")

# %%
if __name__ == "__main__":
    data_root = "data"  
    run_txt_dir = "run_txt"

    if not os.path.exists(run_txt_dir):
        os.makedirs(run_txt_dir)

    splits = ['train', 'test']
    subfolders = ['audio', 'f0', 'f0_norm']

    for split in splits:
        for subfolder in subfolders:
            if subfolder == 'audio':
                output_filename = f"{split}_wav.txt"
            else:
                output_filename = f"{split}_{subfolder}.txt"
            output_file = os.path.join(run_txt_dir, output_filename)

            generate_filelist(data_root, split=split, subfolder=subfolder, output_file=output_file)
    for split in splits:
        filter_common_files(run_txt_dir, split)
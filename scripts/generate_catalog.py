import os
import json
import hashlib
from datetime import datetime, timezone

# 忽略的系统目录与非资料文件
IGNORE_DIRS = {'.git', '.github', 'scripts', 'web', '__pycache__', 'node_modules'}
IGNORE_FILES = {'catalog.json', 'README.md', 'LICENSE', '.gitignore', '.gitattributes', '.DS_Store', 'Thumbs.db'}

def format_size(size_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != 'B' else f"{size_bytes} B"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def calc_sha256(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def build_tree(current_dir: str, rel_prefix: str = "", flat_files: list = None, all_dirs: list = None) -> dict:
    """递归构建任意层级的目录树，并收集扁平文件索引与目录列表"""
    node_dirs = []
    node_files = []
    total_files_count = 0

    try:
        entries = sorted(os.listdir(current_dir))
    except PermissionError:
        entries = []

    for entry in entries:
        if entry.startswith('.'):
            continue

        full_path = os.path.join(current_dir, entry)
        rel_path = f"{rel_prefix}/{entry}" if rel_prefix else entry

        if os.path.isdir(full_path):
            if entry in IGNORE_DIRS:
                continue
            all_dirs.append(rel_path)
            sub_tree = build_tree(full_path, rel_path, flat_files, all_dirs)
            # 仅收录非空文件夹（避免空文件夹干扰展示，如需保留空文件夹可去掉此判断）
            if sub_tree['total_files'] > 0:
                node_dirs.append({
                    'name': entry,
                    'path': rel_path,
                    'total_files': sub_tree['total_files'],
                    'dirs': sub_tree['dirs'],
                    'files': sub_tree['files']
                })
                total_files_count += sub_tree['total_files']

        elif os.path.isfile(full_path):
            if entry in IGNORE_FILES:
                continue
            stat = os.stat(full_path)
            ext = os.path.splitext(entry)[1].lower().lstrip('.') or 'file'
            file_hash = calc_sha256(full_path)
            path_parts = rel_path.split('/')

            file_info = {
                'name': entry,
                'path': rel_path,
                'parts': path_parts,
                'ext': ext,
                'size_bytes': stat.st_size,
                'size_str': format_size(stat.st_size),
                'sha256': file_hash
            }
            node_files.append(file_info)
            flat_files.append(file_info)
            total_files_count += 1

    return {
        'total_files': total_files_count,
        'dirs': node_dirs,
        'files': node_files
    }

def main():
    flat_files = []
    all_dirs = []
    root_tree = build_tree('.', '', flat_files, all_dirs)

    catalog = {
        'updated_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
        'total_files': root_tree['total_files'],
        'directories': sorted(all_dirs),
        'tree': root_tree,
        'files': flat_files
    }

    with open('catalog.json', 'w', encoding='utf-8') as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    print(f"扫描完成：共索引 {len(all_dirs)} 个目录，{root_tree['total_files']} 个文件。")

if __name__ == '__main__':
    main()
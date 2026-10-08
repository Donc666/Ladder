import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

SKIP_DIR_NAMES = {".cache", ".git"}


def _should_skip_dir(path: str) -> bool:
    parts = Path(path).parts
    return any(part in SKIP_DIR_NAMES for part in parts)


def safe_rename_all_paths(root_folder: str):
    """
    把 root 下所有 文件名/目录名 的 '-' 改成 '_'
    bottom-up 遍历，避免父目录先改名导致子路径失效
    """
    n_file = n_dir = n_skip = 0
    for cur, dirs, files in os.walk(root_folder, topdown=False):
        if _should_skip_dir(cur):
            continue
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
        for fn in files:
            if "-" not in fn:
                continue
            old = os.path.join(cur, fn)
            new = os.path.join(cur, fn.replace("-", "_"))
            if os.path.exists(new):
                n_skip += 1
                continue
            os.rename(old, new)
            n_file += 1

        for dn in dirs:
            if "-" not in dn:
                continue
            old = os.path.join(cur, dn)
            new = os.path.join(cur, dn.replace("-", "_"))
            if os.path.exists(new):
                n_skip += 1
                continue
            os.rename(old, new)
            n_dir += 1
    print(f"[rename] files={n_file} dirs={n_dir} skipped_conflicts={n_skip}")
    return n_file, n_dir, n_skip


def patch_urdf_hyphen_only_in_identifiers(urdf_path: str):
    """
    只改 URDF 中“名称/路径”字段:
      - name="..."
      - link="..."
      - joint="..."
      - filename="..."
    不改 xyz/rpy/lower/upper 等数值字段，所以负号不受影响。
    """
    text = Path(urdf_path).read_text(encoding="utf-8", errors="ignore")
    orig = text

    # 只在这些属性的引号内容里替换 '-' -> '_'
    def repl_attr(m):
        key = m.group(1)
        val = m.group(2)
        return f'{key}="{val.replace("-", "_")}"'

    text = re.sub(r'\b(name|link|joint|filename)="([^"]*)"', repl_attr, text)

    if text != orig:
        Path(urdf_path).write_text(text, encoding="utf-8")
        return True
    return False


def patch_mesh_sidecars(root_folder: str):
    """Keep OBJ/MTL references in sync after '-' -> '_' file renames."""
    n_obj = n_mtl = 0
    obj_pat = re.compile(r"^(\s*(?:mtllib|map_\w+|bump|disp|decal)\s+)(\S+)", re.I)
    for cur, dirs, files in os.walk(root_folder):
        if _should_skip_dir(cur):
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
        for fn in files:
            path = Path(cur) / fn
            suf = path.suffix.lower()
            if suf not in {".obj", ".mtl"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            orig = text
            if suf == ".obj":
                # Replace only the mtllib filename so the trailing newline is kept.
                # Reconstructing the line with `$` would drop `\n` and glue `mtllib` onto `o`/`v`.
                text = re.sub(
                    r"(?m)^(\s*mtllib\s+)(\S+)",
                    lambda m: m.group(1) + m.group(2).replace("-", "_"),
                    text,
                )
            else:
                lines = []
                for line in text.splitlines(keepends=True):
                    m = obj_pat.match(line)
                    if m and "-" in m.group(2):
                        rest = line[m.end() :]
                        lines.append(f"{m.group(1)}{m.group(2).replace('-', '_')}{rest}")
                    else:
                        lines.append(line)
                text = "".join(lines)
            if text != orig:
                path.write_text(text, encoding="utf-8")
                if suf == ".obj":
                    n_obj += 1
                else:
                    n_mtl += 1
    print(f"[mesh] patched obj={n_obj} mtl={n_mtl}")


def patch_all_urdfs(root_folder: str):
    scanned = patched = 0
    for cur, dirs, files in os.walk(root_folder):
        if _should_skip_dir(cur):
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
        for fn in files:
            if fn.lower().endswith(".urdf"):
                scanned += 1
                if patch_urdf_hyphen_only_in_identifiers(os.path.join(cur, fn)):
                    patched += 1
    print(f"[urdf] scanned={scanned} patched={patched}")
    return scanned, patched


def uniquify_urdf_visual_names(urdf_path: str) -> bool:
    """PartNet repeats visual name='translation_1' on one link; USD then merges meshes badly."""
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
    except Exception:
        return False
    seen: dict[str, int] = {}
    changed = False
    for vis in root.iter("visual"):
        name = vis.attrib.get("name")
        if not name:
            continue
        n = seen.get(name, 0)
        if n:
            vis.set("name", f"{name}_{n}")
            changed = True
        seen[name] = n + 1
    if changed:
        tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    return changed


def uniquify_all_urdf_visual_names(root_folder: str):
    n = 0
    for cur, dirs, files in os.walk(root_folder):
        if _should_skip_dir(cur):
            dirs[:] = []
            continue
        for fn in files:
            if fn.lower().endswith(".urdf") and uniquify_urdf_visual_names(os.path.join(cur, fn)):
                n += 1
    print(f"[urdf] uniquified duplicate visual names in {n} files")


_GLUED_V = re.compile(
    r"(v\s+[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
    r"\s+[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
    r"\s+[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)"
    r"(?=v\s)"
)


def repair_obj_topology(path: str) -> bool:
    """Fix OBJ issues that make Omniverse drop geometry.

    - `o #ID44` is parsed as a comment, so the object name/group vanishes
    - SSTK sometimes glues two vertex commands on one line: `v x y zv x y z`
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    orig = text
    text = re.sub(r"(?m)^o\s+#", "o ", text)
    text = _GLUED_V.sub(r"\1\n", text)
    if text == orig:
        return False
    Path(path).write_text(text, encoding="utf-8")
    return True


def repair_all_obj_topology(root_folder: str):
    n = 0
    scanned = 0
    for cur, dirs, files in os.walk(root_folder):
        if _should_skip_dir(cur):
            dirs[:] = []
            continue
        for fn in files:
            if not fn.lower().endswith(".obj"):
                continue
            scanned += 1
            if repair_obj_topology(os.path.join(cur, fn)):
                n += 1
    print(f"[obj] repaired topology in {n}/{scanned} files")


if __name__ == "__main__":
    root = os.environ.get("PARTNET_MOBILITY_ROOT", str(Path.home() / "PartNetMobility"))

    # 1) 先改实际文件/目录名
    safe_rename_all_paths(root)

    # 2) 再修 URDF 内引用与 name/link
    patch_all_urdfs(root)

    # 3) OBJ/MTL 里的贴图与 mtllib 路径同步改名
    patch_mesh_sidecars(root)

    # 4) Hydra 丢网格：去掉 o # 注释名、拆开粘连的 v 行、visual 重名
    repair_all_obj_topology(root)
    uniquify_all_urdf_visual_names(root)

    print("完成：文件名 '-'->'_'，URDF/OBJ/MTL 引用已对齐，数值负号保持不变。")
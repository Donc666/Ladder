"""Box: open the lid, or pull the drawer.

Regular_Cover and Fourfold_Cover are both hinged on Cuboidal_Body. The link
origin is the hinge, so the grasp is the cover center: half the cover width
out from the hinge, rotated by the current hinge angle.
"""

import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from my_utils import wxyz_to_xyzw, xyzw_to_wxyz
from partnet_skill_common import (
    CONCEPT_PATHS,
    PartNetConcept,
    PartNetSkill,
    SkillSpec,
    _closed_open_limits,
    _read_joint_q,
    detect_xy_mirror,
    fk_link_world_pos,
    joint_limits,
    map_concept_dir_to_real,
    map_concept_point_to_real,
    pick_grasp_link,
    scene_asset,
    wp,
    xy_in_root_frame,
)

SPEC = SkillSpec(
    type_name="box",
    category="Box",
    grasp_keywords=["cover", "lid", "drawer"],
    motion="revolute",
    scale=(0.4, 0.4, 0.4),
    spawn_z=0.38,
    concept_xy=(1.45, 0.0),
    concept_z_drop=0.0,
    max_deg=30,
)


def _concept_frame_lift(urdf_path: str, scale_z: float) -> float:
    """Concept root is Rx(90) plus a z origin. PartNet boxes have origin 0."""
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return 0.0
    for joint in root.findall("joint"):
        if not (joint.attrib.get("name") or "").startswith("world_to_"):
            continue
        origin = joint.find("origin")
        if origin is None or not origin.attrib.get("xyz"):
            return 0.0
        try:
            z = float(origin.attrib["xyz"].split()[2])
        except (TypeError, ValueError):
            return 0.0
        return z * float(scale_z)
    return 0.0
DRAWER = replace(SPEC, grasp_keywords=["drawer"], motion="prismatic", scale=(0.22, 0.22, 0.22), spawn_z=0.38)


def _mesh_vertices(urdf_path: str, link_name: str) -> np.ndarray:
    root = ET.parse(urdf_path).getroot()
    for link in root.findall("link"):
        if link.attrib.get("name") != link_name:
            continue
        mesh = link.find("visual/geometry/mesh")
        if mesh is None or not mesh.attrib.get("filename"):
            break
        path = Path(urdf_path).parent / mesh.attrib["filename"]
        verts = []
        with path.open() as handle:
            for line in handle:
                if line.startswith("v "):
                    verts.append([float(x) for x in line.split()[1:4]])
        if verts:
            return np.asarray(verts, dtype=np.float64)
    return np.zeros((0, 3), dtype=np.float64)


def _cover_joint(urdf_path: str, link_name: str):
    root = ET.parse(urdf_path).getroot()
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is None or child.attrib.get("link") != link_name:
            continue
        if (joint.attrib.get("type") or "").lower() not in {"revolute", "continuous"}:
            continue
        axis = joint.find("axis")
        raw = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if axis is not None and axis.attrib.get("xyz"):
            raw = np.array([float(x) for x in axis.attrib["xyz"].split()], dtype=np.float64)
        n = float(np.linalg.norm(raw))
        if n < 1e-8:
            raw = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            raw = raw / n
        return joint.attrib.get("name") or "", raw
    return None, None


def flap_center_local(urdf_path: str, link_name: str):
    """Cover center in the link frame: half the width from the hinge.

    The joint axis is the hinge. Width is the mesh extent perpendicular to
    that axis, from the hinge out to the free edge.
    """
    axis = _cover_joint(urdf_path, link_name)[1]
    verts = _mesh_vertices(urdf_path, link_name)
    if axis is None or len(verts) == 0:
        return None
    along = verts @ axis
    radial = verts - along[:, None] * axis
    outward = radial.mean(axis=0)
    n = float(np.linalg.norm(outward))
    if n < 1e-8:
        farthest = int(np.argmax(np.linalg.norm(radial, axis=1)))
        outward = radial[farthest]
        n = float(np.linalg.norm(outward))
    if n < 1e-8:
        return None
    outward = outward / n
    width = float(np.max(radial @ outward))
    center = outward * (0.5 * width) + axis * float(along.mean())
    normal = np.cross(axis, outward)
    nn = float(np.linalg.norm(normal))
    if nn < 1e-8:
        return None
    normal = normal / nn
    return {
        "center": center,
        "axis": axis,
        "outward": outward,
        "normal": normal,
        "width": width,
    }


def _unit(v):
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return None
    return np.asarray(v, dtype=np.float64) / n


def _hand_axes(outward, normal):
    """Fingers point toward the hinge. The jaw closes through the flap thickness."""
    z_axis = _unit(-np.asarray(outward, dtype=np.float64))
    if z_axis is None:
        return None
    y_axis = np.asarray(normal, dtype=np.float64)
    y_axis = y_axis - np.dot(y_axis, z_axis) * z_axis
    y_axis = _unit(y_axis)
    if y_axis is None:
        return None
    x_axis = _unit(np.cross(y_axis, z_axis))
    if x_axis is None:
        return None
    y_axis = np.cross(z_axis, x_axis)
    return x_axis, y_axis, z_axis


def _quat_from_axes(axes):
    rot = np.column_stack(axes)
    return xyzw_to_wxyz(R.from_matrix(rot).as_quat())


class box_concept(PartNetConcept):
    spec = SPEC

    def import2sim(self, urdf_path, instance_name, num_envs):
        cfgs = super().import2sim(urdf_path, instance_name, num_envs)
        if not cfgs:
            return cfgs
        lift = _concept_frame_lift(self.concept_urdf_path, self.scale[2])
        x, y = self.spec.spawn_xy
        pos = (float(x) + 0.75, float(y), float(self.spec.spawn_z) - lift)
        cfgs[0].pos = pos
        print(
            f"[box_concept] beside the box at {tuple(round(v, 3) for v in pos)} "
            f"frame_lift={lift:.3f}"
        )
        return cfgs


class box(PartNetSkill):
    spec = SPEC

    def open(self, scene, env_id=0, fraction=0.5):
        concept_urdf = CONCEPT_PATHS.get(self.spec.type_name) or getattr(self, "concept_urdf_path", None)
        if concept_urdf and _cover_links(concept_urdf):
            pack = self._open_cover(scene, env_id, fraction, concept_urdf)
            if pack:
                return pack
        return self.execute(scene, env_id, fraction, spec_override=SPEC)

    def _open_cover(self, scene, env_id, fraction, concept_urdf):
        spec = self.spec
        asset = scene_asset(scene, self.scene_key, env_id)
        if asset is None:
            return None
        try:
            my_concept = scene[f"my_concept_{spec.type_name}__env{env_id}"]
        except (KeyError, AttributeError):
            return None
        names = _cover_links(concept_urdf)
        if not names:
            return None
        link_name = pick_grasp_link(my_concept, concept_urdf, ["cover"], index=0)
        if link_name not in names:
            link_name = names[0]
        joint_name, _axis = _cover_joint(concept_urdf, link_name)
        geom = flap_center_local(concept_urdf, link_name)
        if not joint_name or geom is None:
            return None

        lo, hi = joint_limits(concept_urdf, joint_name)
        q_closed, q_opened = _closed_open_limits(lo, hi)
        q_now = _read_joint_q(my_concept, joint_name, env_id)
        if q_now is None:
            q_now = float(q_closed)
        frac = float(np.clip(fraction, 0.05, 1.0))
        delta = (float(q_opened) - float(q_closed)) * frac
        step = np.deg2rad(10.0)
        n_steps = max(1, int(round(abs(delta) / step)))
        angles = [float(q_now) + delta * (i / n_steps) for i in range(n_steps + 1)]

        concept_root = my_concept.data.root_pos_w[env_id].detach().cpu().numpy()
        concept_quat = my_concept.data.root_quat_w[env_id].detach().cpu().numpy()
        real_root = asset.data.root_pos_w[env_id].detach().cpu().numpy()
        real_quat = asset.data.root_quat_w[env_id].detach().cpu().numpy()
        scale = np.asarray(self.scale, dtype=np.float64)

        def at_angle(q):
            hinge, quat = fk_link_world_pos(
                concept_urdf, link_name, {joint_name: float(q)}, concept_root, concept_quat, scale
            )
            rot = R.from_quat(wxyz_to_xyzw(quat))
            center = np.asarray(hinge, dtype=np.float64) + rot.apply(geom["center"] * scale)
            outward = rot.apply(geom["outward"])
            normal = rot.apply(geom["normal"])
            return hinge, center, outward, normal

        hinge0, center0, outward0, normal0 = at_angle(angles[0])
        c_xy = xy_in_root_frame(my_concept, center0, env_id)
        r_xy = xy_in_root_frame(asset, real_root, env_id)
        mirror = detect_xy_mirror(c_xy, r_xy)

        def to_real_point(point):
            return map_concept_point_to_real(point, concept_root, concept_quat, real_root, real_quat, mirror)

        def to_real_dir(direction):
            return map_concept_dir_to_real(direction, concept_quat, real_quat, mirror)

        poses = []
        for q in angles:
            _hinge, center, outward, normal = at_angle(q)
            outward_w = to_real_dir(outward)
            normal_w = to_real_dir(normal)
            axes = _hand_axes(outward_w, normal_w)
            if axes is None:
                return None
            contact = to_real_point(center)
            # panda_hand +Z points along the fingers. grasp_z is negative, so the palm sits behind the flap.
            palm = contact + R.from_matrix(np.column_stack(axes)).apply([0.0, 0.0, spec.grasp_z])
            poses.append((palm, _quat_from_axes(axes), contact))

        palm0, q0, contact0 = poses[0]
        r_hand = R.from_quat(wxyz_to_xyzw(q0))
        approach = palm0 + r_hand.apply([0.0, 0.0, spec.approach_z - spec.grasp_z])
        last_palm, last_q, _last_contact = poses[-1]
        r_last = R.from_quat(wxyz_to_xyzw(last_q))
        retreat = last_palm + r_last.apply([0.0, 0.0, spec.retreat_z])
        executor = "robot_left" if float(contact0[1]) > 0.0 else "robot_right"
        width_m = float(geom["width"] * scale[0])
        print(
            f"[box] {link_name} hinge=({hinge0[0]:.3f},{hinge0[1]:.3f},{hinge0[2]:.3f}) "
            f"width={width_m:.3f} m q={q_now:.3f}->{angles[-1]:.3f} rad "
            f"contact=({contact0[0]:.3f},{contact0[1]:.3f},{contact0[2]:.3f}) executor={executor}"
        )
        pack = [
            wp(approach, q0, "open", executor),
            wp(palm0, q0, "open", executor),
            *[wp(palm, quat, "close", executor) for palm, quat, _contact in poses[1:]],
            wp(last_palm, last_q, "open", executor),
            wp(retreat, last_q, "open", executor),
        ]
        return pack

    def pull(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction, spec_override=DRAWER)


def _cover_links(urdf_path: str) -> list[str]:
    """Hinged lids. Fourfold flaps stay in id order; a single Regular_Cover is the other kind."""
    root = ET.parse(urdf_path).getroot()
    fourfold = []
    regular = []
    for link in root.findall("link"):
        name = link.attrib.get("name") or ""
        low = name.lower()
        if low.startswith("fourfold_cover"):
            fourfold.append(name)
        elif low == "regular_cover":
            regular.append(name)
    if fourfold:
        return sorted(fourfold)
    return regular


ATOMS = {"open", "pull"}

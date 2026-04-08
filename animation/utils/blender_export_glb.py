import argparse
import os
from collections import defaultdict

import bpy
import numpy as np


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.armatures:
        if block.users == 0:
            bpy.data.armatures.remove(block)


def parse_rig_txt(rig_path):
    joints = []
    joint_names = []
    name_to_pos = {}
    parents = {}
    children = defaultdict(list)
    root_name = None
    skinning = defaultdict(list)  # vertex_idx -> [(joint_name, weight)]

    with open(rig_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for line in lines:
        parts = line.split()
        head = parts[0]
        if head == "joints":
            name = parts[1]
            pos = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float32)
            joint_names.append(name)
            joints.append(pos)
            name_to_pos[name] = pos
        elif head == "root":
            root_name = parts[1]
        elif head == "hier":
            p, c = parts[1], parts[2]
            parents[c] = p
            children[p].append(c)
        elif head == "skin":
            v_idx = int(parts[1])
            for i in range(2, len(parts), 2):
                if i + 1 < len(parts):
                    skinning[v_idx].append((parts[i], float(parts[i + 1])))

    if root_name is None:
        raise RuntimeError("root joint not found in rig file")

    return joint_names, name_to_pos, parents, children, root_name, skinning


def import_mesh_obj(mesh_path):
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=mesh_path)
    else:
        bpy.ops.import_scene.obj(filepath=mesh_path)

    imported_meshes = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not imported_meshes:
        raise RuntimeError("No mesh imported from OBJ")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in imported_meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported_meshes[0]
    if len(imported_meshes) > 1:
        bpy.ops.object.join()
    mesh_obj = bpy.context.view_layer.objects.active
    return mesh_obj


def build_armature(joint_names, name_to_pos, parents, children):
    bpy.ops.object.armature_add(enter_editmode=True, location=(0.0, 0.0, 0.0))
    arm_obj = bpy.context.object
    arm_obj.name = "RigArmature"
    arm_data = arm_obj.data
    arm_data.name = "RigArmatureData"

    # Remove default bone.
    if arm_data.edit_bones:
        arm_data.edit_bones.remove(arm_data.edit_bones[0])

    # A small scale-aware tail offset for leaf bones.
    all_pos = np.stack([name_to_pos[n] for n in joint_names], axis=0)
    bbox = all_pos.max(axis=0) - all_pos.min(axis=0)
    tail_eps = max(float(np.max(bbox)) * 0.02, 1e-4)

    created = {}
    for name in joint_names:
        bone = arm_data.edit_bones.new(name)
        head = name_to_pos[name]
        if children.get(name):
            child_head = name_to_pos[children[name][0]]
            tail = child_head
            if np.linalg.norm(tail - head) < 1e-8:
                tail = head + np.array([0.0, 0.0, tail_eps], dtype=np.float32)
        else:
            tail = head + np.array([0.0, 0.0, tail_eps], dtype=np.float32)

        bone.head = tuple(float(v) for v in head)
        bone.tail = tuple(float(v) for v in tail)
        created[name] = bone

    # Set hierarchy in edit mode.
    for child, parent in parents.items():
        if child in created and parent in created:
            created[child].parent = created[parent]
            created[child].use_connect = False

    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def bind_weights(mesh_obj, arm_obj, joint_names, skinning):
    # Create vertex groups for all joints.
    for name in joint_names:
        mesh_obj.vertex_groups.new(name=name)

    # Assign skin weights.
    for v_idx, jw_list in skinning.items():
        for j_name, w in jw_list:
            vg = mesh_obj.vertex_groups.get(j_name)
            if vg is not None:
                vg.add([v_idx], float(w), "REPLACE")

    # Add armature modifier.
    mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
    mod.object = arm_obj

    # Parent mesh to armature.
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type="ARMATURE")


def compute_mesh_scale(mesh_obj):
    coords = np.array([mesh_obj.matrix_world @ v.co for v in mesh_obj.data.vertices], dtype=np.float32)
    if coords.size == 0:
        return 1.0
    ext = coords.max(axis=0) - coords.min(axis=0)
    scale = float(np.max(ext))
    return scale if scale > 1e-8 else 1.0


def apply_animation(
    arm_obj,
    joint_names,
    local_quats,
    root_quats,
    root_pos,
    apply_root_motion,
    root_pos_scale,
):
    T, J, C = local_quats.shape
    if C != 4:
        raise RuntimeError(f"local_quats shape expected (...,4), got {local_quats.shape}")
    if J != len(joint_names):
        raise RuntimeError(f"Joint count mismatch: quats J={J}, rig joints={len(joint_names)}")

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = int(T)

    # Ensure bones use quaternion mode.
    for name in joint_names:
        pb = arm_obj.pose.bones.get(name)
        if pb is not None:
            pb.rotation_mode = "QUATERNION"

    arm_obj.rotation_mode = "QUATERNION"

    for t in range(T):
        frame = t + 1
        bpy.context.scene.frame_set(frame)

        # Local joint rotations.
        for j, name in enumerate(joint_names):
            pb = arm_obj.pose.bones.get(name)
            if pb is None:
                continue
            q = local_quats[t, j]
            pb.rotation_quaternion = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)

        if apply_root_motion:
            rq = root_quats[t]
            rp = root_pos[t] * root_pos_scale
            arm_obj.rotation_quaternion = (float(rq[0]), float(rq[1]), float(rq[2]), float(rq[3]))
            arm_obj.location = (float(rp[0]), float(rp[1]), float(rp[2]))
            arm_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            arm_obj.keyframe_insert(data_path="location", frame=frame)


def export_glb(out_glb):
    out_dir = os.path.dirname(out_glb)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=out_glb,
        export_format="GLB",
        export_animations=True,
        use_selection=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Export Puppeteer animation result to GLB via Blender")
    parser.add_argument("--mesh_obj", required=True)
    parser.add_argument("--rig_txt", required=True)
    parser.add_argument("--local_quats_npy", required=True)
    parser.add_argument("--root_quats_npy", required=True)
    parser.add_argument("--root_pos_npy", required=True)
    parser.add_argument("--out_glb", required=True)
    parser.add_argument("--apply_root_motion", type=int, default=1)

    argv = []
    if "--" in bpy.sys.argv:
        argv = bpy.sys.argv[bpy.sys.argv.index("--") + 1 :]
    return parser.parse_args(argv)


def main():
    args = parse_args()
    clear_scene()

    local_quats = np.load(args.local_quats_npy)
    root_quats = np.load(args.root_quats_npy)
    root_pos = np.load(args.root_pos_npy)

    joint_names, name_to_pos, parents, children, _root_name, skinning = parse_rig_txt(args.rig_txt)
    mesh_obj = import_mesh_obj(args.mesh_obj)
    arm_obj = build_armature(joint_names, name_to_pos, parents, children)
    bind_weights(mesh_obj, arm_obj, joint_names, skinning)

    root_scale = compute_mesh_scale(mesh_obj)
    apply_animation(
        arm_obj=arm_obj,
        joint_names=joint_names,
        local_quats=local_quats,
        root_quats=root_quats,
        root_pos=root_pos,
        apply_root_motion=bool(args.apply_root_motion),
        root_pos_scale=root_scale,
    )
    export_glb(args.out_glb)
    print(f"GLB exported: {args.out_glb}")


if __name__ == "__main__":
    main()

import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer
import numpy as np
import argparse
import os

# -------------------------------------------------------
#                  XML MERGE FUNCTION
# -------------------------------------------------------
def merge_xml(scene_xml, drone_xml, output_xml="merged_env.xml"):
    treeA = ET.parse(scene_xml)
    treeB = ET.parse(drone_xml)

    rootA = treeA.getroot()
    rootB = treeB.getroot()

    # New root
    merged = ET.Element("mujoco", model="merged_env")

    # ---------------- ASSET ----------------
    merged_asset = ET.SubElement(merged, "asset")

    for root in (rootA, rootB):
        asset = root.find("asset")
        if asset is not None:
            for child in list(asset):
                merged_asset.append(child)

    # ---------------- WORLDBODY ----------------
    merged_world = ET.SubElement(merged, "worldbody")

    for root in (rootA, rootB):
        world = root.find("worldbody")
        if world is not None:
            for child in list(world):
                merged_world.append(child)

    # ---------------- ACTUATOR ----------------
    merged_actu = None
    for root in (rootA, rootB):
        actu = root.find("actuator")
        if actu is not None:
            if merged_actu is None:
                merged_actu = ET.SubElement(merged, "actuator")
            for child in list(actu):
                merged_actu.append(child)

    # ---------------- SENSOR ----------------
    merged_sensor = None
    for root in (rootA, rootB):
        sens = root.find("sensor")
        if sens is not None:
            if merged_sensor is None:
                merged_sensor = ET.SubElement(merged, "sensor")
            for child in list(sens):
                merged_sensor.append(child)

    # Save merged XML
    ET.ElementTree(merged).write(output_xml)
    print(f"[OK] Merged XML written to: {output_xml}")
    return output_xml


# -------------------------------------------------------
#               LOAD + RENDER THE ENVIRONMENT
# -------------------------------------------------------
def run_mujoco(xml_path):
    print(f"[INFO] Loading model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Print available cameras
    cam_ids = [i for i in range(model.ncam)]
    cam_names = [model.camera(i).name for i in cam_ids]

    print("[INFO] Cameras:", list(zip(cam_ids, cam_names)))

    # Launch viewer
    print("[OK] Running viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:

        # OPTIONAL: render from the drone camera
        # (this sets the *viewer* camera to follow the drone_cam)
        try:
            drone_cam_id = model.camera("drone_cam").id
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = drone_cam_id
            print("[INFO] Using drone_cam for display")
        except:
            print("[WARN] drone_cam not found, using free camera")

        # Physics loop
        while viewer.is_running():
            # Simple hover (optional)
            if model.nu == 4:
                data.ctrl[:] = np.array([6, 6, 6, 6])

            mujoco.mj_step(model, data)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_xml", help="path to background scene XML", default='office_0_quantized_16/office_0.xml')
    parser.add_argument("--drone_xml", help="path to drone XML", default='drone_with_camera.xml')
    parser.add_argument("--out", default="office_0_quantized_16/merged_env.xml", help="output XML file")
    args = parser.parse_args()

    merged_path = merge_xml(args.scene_xml, args.drone_xml, args.out)
    run_mujoco(merged_path)

from pxr import Usd, UsdGeom, Gf

import sys

angle_y = float(sys.argv[1])
# angle_x = float(sys.argv[1])
# angle_y = float(sys.argv[2])
# angle_z = float(sys.argv[3])

# target_angle = Gf.Vec3f(angle_x, angle_y, angle_z)

target_angle = Gf.Vec3f(0.0, angle_y, 0.0) 
# def set_plane_angle(angle: tuple = (0.0, 0.0, 0.0)):
# --- 1. Define file paths and prim path ---
input_file_path = "/home/workstation/pavaris_ws/isaaclab-locomotion-learning/source/PlasticNeuralNet/PlasticNeuralNet/assets/models/sloped_plane.usd"
output_file_path = input_file_path
prim_path_to_edit = "/Root/FlatGrid"

# --- 2. Open the existing USD stage ---
stage = Usd.Stage.Open(input_file_path)

# --- 3. Get the prim you want to edit ---
flat_grid_prim = stage.GetPrimAtPath(prim_path_to_edit)

# Check if the prim exists and is valid
if not flat_grid_prim:
    print(f"Error: Prim at path '{prim_path_to_edit}' not found.")
else:
    # --- 4. Get the Xformable schema for the prim ---
    # This gives us access to its transformation properties
    xformable = UsdGeom.Xformable(flat_grid_prim)

    xform_ops = xformable.GetOrderedXformOps()

    rotation_op = xformable.GetRotateXYZOp()

    new_rotation_in_degrees = target_angle
    
    rotation_op.Set(new_rotation_in_degrees)
    
    print(f"Successfully set rotation for '{prim_path_to_edit}' to {new_rotation_in_degrees}")

    # --- 6. Save the modified stage to a new file ---
    stage.Save() # This will save the changes back to the original file

    print(f"Modified stage saved to '{output_file_path}'")



# # def set_plane_angle(angle: tuple = (0.0, 0.0, 0.0)):
# # --- 1. Define file paths and prim path ---
# input_file_path = "/home/nasree-hsm/workspace/isaaclab-locomotion-learning/source/PlasticNeuralNet/PlasticNeuralNet/assets/models/slalom_bendbody_19dof.usd"
# output_file_path = input_file_path
# prim_path_to_edit = "/gecko_base"

# # --- 2. Open the existing USD stage ---
# stage = Usd.Stage.Open(input_file_path)


# # --- 3. Get the prim you want to edit ---
# flat_grid_prim = stage.GetPrimAtPath(prim_path_to_edit)


# # Check if the prim exists and is valid
# if not flat_grid_prim:
#     print(f"Error: Prim at path '{prim_path_to_edit}' not found.")
# else:
#     # --- 4. Get the Xformable schema for the prim ---
#     # This gives us access to its transformation properties
#     xformable = UsdGeom.Xformable(flat_grid_prim)

#     # rtranslation_op = xformable.GetTranslateZOp()

#     rotation_op = xformable.GetRotateXYZOp()


#     new_rotation_in_degrees = target_angle 
#     # new_rotation_in_degrees = Gf.Vec3f(angle_x, angle_y, angle_z)
    
#     rotation_op.Set(new_rotation_in_degrees)
    
#     print(f"Successfully set rotation for '{prim_path_to_edit}' to {new_rotation_in_degrees}")

#     # --- 6. Save the modified stage to a new file ---
#     stage.Save() # This will save the changes back to the original file

#     print(f"Modified stage saved to '{output_file_path}'")
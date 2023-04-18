import numpy as np
import torch
from matplotlib import pyplot as plt
from .quaternion import *


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def pyplot_skeleton(topology, positions, rel=False, ax=None, show=True, color='black'):
    # Plot a skeleton in 3d
    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')

    abs_positions = torch.zeros_like(positions)
   
    for j, i in enumerate(topology):

        if rel:
            abs_positions[j] = abs_positions[i] + positions[j]
        else:
            abs_positions[j] = positions[j]

        if j>0:
            ax.plot(
                np.array(abs_positions[[i,j], 0]),
                np.array(abs_positions[[i,j], 1]),
                np.array(abs_positions[[i,j], 2]),
                color=color
            )
   
    ax.scatter(
        np.array(abs_positions[:, 0]),
        np.array(abs_positions[:, 1]),
        np.array(abs_positions[:, 2]),
        color=color,
        marker='o'
    )

    if show:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        set_axes_equal(ax)
        plt.show()
    else:
        return ax


def find_quaternions(topology, offsets, positions):
    """
    Compute the quaternions corresponding to the absolute joint rotations. For J joints, J-1 quaternions are computed.
    The quaternions for the first joint (the base) are set to the identity rotation quaternion.
    Parameters:
        topology: python list providing the joint topology
        offsets: Tensor of shape (batch_size, num_joints, 3)
        positions: Tensor of shape (batch_size, num_joints, 3)
    Returns:
        quaternion: Tensor of shape (batch_size, num_joints, 4)
    """

    batch_size = positions.shape[0]
    #rel_quaternion = torch.zeros(batch_size, len(topology), 4)
    #rel_quaternion[:, 0] = torch.Tensor([1., 0., 0., 0.])
    quaternion = torch.zeros(batch_size, len(topology), 4)
    quaternion[:, 0] = torch.Tensor([1., 0., 0., 0.])

    for joint, parent in enumerate(topology):

        if joint==0:
            continue

        # Compute the relative position
        delta = positions[:, joint] - positions[:, parent]

        # Compute the axis of rotation between the offset and the relative position
        u = torch.linalg.cross(offsets[:, joint], delta, axis=-1)

        # Return identity rotation when offset or delta is null
        # quaternion[torch.norm(u, axis=-1) == 0., joint] = torch.Tensor([1., 0., 0., 0.])
        den = torch.linalg.norm(delta, dim=-1) * torch.linalg.norm(offsets[:, joint], dim=-1)

        sin_alpha = torch.linalg.norm(u, dim=-1) / den
        cos_alpha = torch.sum(delta * offsets[:, joint], axis=-1) / den
        alpha = torch.atan2(sin_alpha, cos_alpha)

        # Normalize u
        u = u / torch.linalg.norm(u, dim=-1, keepdim=True)

        # Define the corresponding quaternion
        quaternion[:, joint, 0] = torch.cos(alpha/2)
        quaternion[:, joint, 1] = torch.sin(alpha/2) * u[:, 0]
        quaternion[:, joint, 2] = torch.sin(alpha/2) * u[:, 1]
        quaternion[:, joint, 3] = torch.sin(alpha/2) * u[:, 2]
       
        # Compute the relative quaternion
        # rel_quaternion[joint] = quat_left_multiply(quat_inverse(quaternion[parent]), quaternion[joint])

    return quaternion #, rel_quaternion


def find_rotations(topology, offsets, positions):
    # Find the rotations (in quaternions) for each squeleton joint
    # NB: for convention, we choose to set to 0 the Euler angle gamma

    rel_euler = np.zeros((len(topology), 3))
    quater = np.zeros((len(topology), 4))
    quater[0] = np.array([1., 0., 0., 0.])
    rel_quater = np.zeros_like(quater)
    rel_quater[0] = np.array([1., 0., 0., 0.])

    for j in range(len(topology)):

        if j==0:
            continue

        # Initialize a quaternion encoding the offset
        q_offset = np.zeros(4)

        # Get the offset
        q_offset[1:] = offsets[j]
       
        if np.linalg.norm(q_offset)>0:

            # Normalize to obtain a unitary vector
            q_offset /= np.linalg.norm(q_offset)

            # Compute the euler angles with gamma=0
            alpha = np.arctan2(q_delta[1], -q_delta[2])
            beta = np.arccos(q_delta[3].clip(-1, 1))

        else:
            alpha = 0.
            beta = 0.
       
        # Compute the corresponding quaternion
        q_rot_offset = np.array([
            np.cos(alpha/2)*np.cos(beta/2),
            np.cos(alpha/2)*np.sin(beta/2),
            np.sin(alpha/2)*np.sin(beta/2),
            np.sin(alpha/2)*np.cos(beta/2)
        ])

        # Get the quaternion of its parent
        q_parent = quater[topology[j]]

        # Compose the offset rotation with the parent rotation
        q_reference = quat_left_multiply(q_parent, q_rot_offset)

        # Initialize a quaternion encoding the relative position of the joint
        q_delta = np.zeros(4)

        # Get the relative position of the joint
        q_delta[1:] = positions[j] - positions[topology[j]]
       
        if np.linalg.norm(q_delta)>0:

            # Normalize to obtain a unitary vector
            q_delta /= np.linalg.norm(q_delta)

            # Express this vector in the relative coordinate system of reference
            q_delta = quat_left_multiply(quat_inverse(q_reference), quat_left_multiply(q_delta, q_reference))

            # Compute the euler angles with gamma=0
            alpha = np.arctan2(q_delta[1], -q_delta[2])
            beta = np.arccos(q_delta[3].clip(-1, 1))

        else:
            alpha = 0.
            beta = 0.

        rel_euler[j] = np.array([alpha, beta, 0.])
       
        # Compute the corresponding relative quaternion
        rel_quater[j] = np.array([
            np.cos(alpha/2)*np.cos(beta/2),
            np.cos(alpha/2)*np.sin(beta/2),
            np.sin(alpha/2)*np.sin(beta/2),
            np.sin(alpha/2)*np.cos(beta/2)
        ])

        # Compute the corresponding absolute quaternion
        quater[j] = quat_left_multiply(q_parent, rel_quater[j])

    return rel_euler, quater, rel_quater


def forward_kinematics(topology, base_position, offsets, quaternions, rel=False):
    """
    Computes the positions of each joints given rotations and offsets.
    The rotation quaternions are either relative or absolute.
    Parameters:
        topology: list of shape (num_joints)
        base_position: Tensor of shape (batch_size, 3)
        offsets: Tensor of shape (batch_size, num_joints, 3)
        quaternions: Tensor of shape (batch_size, num_joints, 4)
        rel: boolean indicating whether the given quaternions are computed relative to parent joint orientation or in an absolute frame
    Returns:
        positions: Tensor of shape (batch_size, num_joints, 3) giving the absolute positions of each joint
        rel_positions: Tensor of shape (batch_size, num_joints, 3) giving the relative positions of each joint relative to its parent
    """

    rel_positions = offsets.clone()
    positions = offsets.clone()
    quaternions = quaternions.clone()
   
    # Base position
    rel_positions[:, 0] = base_position
    positions[:, 0] = base_position

    # Compute the joint positions based on topology, offsets and rotations  
    for joint, parent in enumerate(topology):

        # The base position is already set
        if joint==0:
            continue

        # If quaternions are given in the relative frame, it has to be transformed in the absolute frame
        if rel:
            quaternions[:, joint] = batch_quat_left_multiply(quaternions[:, parent], quaternions[:, joint])

        # Rotate the offset given the absolute rotation quaternion
        rel_positions[:, joint] = batch_rotate_vector(offsets[:, joint], quaternions[:, joint])

        # Add the relative position to the parent relative position to get the absolute joint position
        positions[:, joint] = positions[:, parent] + rel_positions[:, joint]

    return positions, rel_positions


def build_adj_matrix(topology, fill_diag=False):
    """
    Computes the adjacency matrix corresponding to a given skeleton topology
    Parameters:
        topology: python list giving the parent of every node
        fill_diag: Boolean specifying whether to fill the diagonal with ones or not
    Returns:
        adj_matrix: torch Tensor of shape (lenght, lenght)
    """
    adj_matrix = torch.zeros(len(topology), len(topology))
    for joint, parent in enumerate(topology):
        if parent > -1:
            adj_matrix[parent, joint] = 1.0

    return adj_matrix

def reorient_skeleton(skeleton, topology):

    batch_size, n_joints, _ = skeleton.shape

    # Measure skeleton bone lengths
    lengths = torch.Tensor(batch_size, n_joints)
    for child, parent in enumerate(topology):
        lengths[:, child] = torch.sqrt(
            torch.sum(
                (skeleton[:, child] - skeleton[:, parent])**2,
                axis=-1
            )
        )
   
    # Find the corresponding angles
    offsets = torch.zeros(batch_size, n_joints, 3)
    offsets[:, :, -1] = lengths
    quaternions = find_quaternions(topology, offsets, skeleton)
   
    # Re-orient according to the pelvis->chest orientation
    base_orientation = quaternions[:, 7:8].repeat(1, n_joints, 1).reshape(batch_size*n_joints, 4)
    base_orientation += 1e-3 * torch.randn_like(base_orientation)
    quaternions = quaternions.reshape(batch_size*n_joints, 4)
    quaternions = batch_quat_left_multiply(
        batch_quat_inverse(base_orientation),
        quaternions
    )
    quaternions = quaternions.reshape(batch_size, n_joints, 4)
   
    # Use these quaternions in the forward kinematics
    new_skeleton = forward_kinematics(
        topology,
        torch.zeros(batch_size, 3),
        offsets,
        quaternions
    )[0]

    # Measure the hip orientation
    alpha = np.arctan2(
        new_skeleton[0, 1, 1] - new_skeleton[0, 0, 1],
        new_skeleton[0, 1, 0] - new_skeleton[0, 0, 0]
    )
   
    # Rotate by alpha around z
    rotation = torch.Tensor([np.cos(alpha/2), 0, 0, np.sin(alpha/2)]).unsqueeze(0).repeat(batch_size*n_joints, 1)
    quaternions = quaternions.reshape(batch_size*n_joints, 4)
    quaternions = batch_quat_left_multiply(
        batch_quat_inverse(rotation),
        quaternions
    )
    quaternions = quaternions.reshape(batch_size, n_joints, 4)
   
    # Use these quaternions in the forward kinematics with the Poppy skeleton
    new_skeleton = forward_kinematics(
        topology,
        torch.zeros(batch_size, 3),
        offsets,
        quaternions
    )[0]

    return new_skeleton


def extract_skeletons(skeletons_A, skeletons_B, alpha, depth_confidence=0.0):

    depth_confidence = 0.0

    w = .5 * depth_confidence

    new_skeletons = torch.zeros_like(skeletons_A)

    new_skeletons[:, :, 0] = (1-w) * skeletons_A[:, :, 0] \
        + w * (np.cos(alpha) * skeletons_A[:, :, 1] - skeletons_B[:, :, 1]) / np.sin(alpha)

    new_skeletons[:, :, 2] = .5 * (skeletons_A[:, :, 2] + skeletons_B[:, :, 2])

    new_skeletons[:, :, 1] = w * skeletons_A[:, :, 1] \
        + (1-w) * (skeletons_B[:, :, 0] - np.cos(alpha) * skeletons_A[:, :, 0]) / np.sin(alpha)

    return new_skeletons
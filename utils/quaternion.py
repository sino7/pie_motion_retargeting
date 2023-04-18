import numpy as np
import torch

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def batch_quat_conjugate(q):
    c = torch.zeros_like(q)
    c[:, 0] = q[:, 0]
    c[:, 1] = -q[:, 1]
    c[:, 2] = -q[:, 2]
    c[:, 3] = -q[:, 3]
    return c
    
def quat_inverse(q):
    q_norm = q/np.linalg.norm(q)
    return quat_conjugate(q_norm)

def batch_quat_inverse(q):
    q_norm = q/torch.linalg.norm(q, dim=1, keepdim=True)
    return batch_quat_conjugate(q_norm)

def quat_left_multiply(l,r):
    q = np.zeros(4)
    q[0] = l[0]*r[0] - l[1]*r[1] - l[2]*r[2] - l[3]*r[3]
    q[1] = l[0]*r[1] + l[1]*r[0] + l[2]*r[3] - l[3]*r[2]
    q[2] = l[0]*r[2] + l[2]*r[0] + l[3]*r[1] - l[1]*r[3]
    q[3] = l[0]*r[3] + l[3]*r[0] + l[1]*r[2] - l[2]*r[1]
    return q

def batch_quat_left_multiply(l, r):
    batch_size, _ = l.shape
    q = torch.zeros(batch_size, 4)
    q[:, 0] = l[:, 0]*r[:, 0] - l[:, 1]*r[:, 1] - l[:, 2]*r[:, 2] - l[:, 3]*r[:, 3]
    q[:, 1] = l[:, 0]*r[:, 1] + l[:, 1]*r[:, 0] + l[:, 2]*r[:, 3] - l[:, 3]*r[:, 2]
    q[:, 2] = l[:, 0]*r[:, 2] + l[:, 2]*r[:, 0] + l[:, 3]*r[:, 1] - l[:, 1]*r[:, 3]
    q[:, 3] = l[:, 0]*r[:, 3] + l[:, 3]*r[:, 0] + l[:, 1]*r[:, 2] - l[:, 2]*r[:, 1]
    return q

def rotate_vector(vector, quaternion):
    # Apply the rotation defined by the quaternion input onto the vector
    q_vector = np.zeros(4)
    q_vector[1:] = vector
    return quat_left_multiply(quaternion, quat_left_multiply(q_vector, quat_inverse(quaternion)))[1:]

def batch_rotate_vector(vector, quaternion):
    batch_size, _ = vector.shape
    q_vector = torch.zeros(batch_size, 4)
    q_vector[:, 1:] = vector
    return batch_quat_left_multiply(quaternion, batch_quat_left_multiply(q_vector, batch_quat_inverse(quaternion)))[:, 1:]

def euler_convention(angles):

    # Beta in [0, pi]
    alpha, beta, gamma = angles
    if beta < 0:
        alpha = np.pi + alpha
        beta = - beta
        gamma = np.pi + gamma

    # Alpha and Gamma in [-pi, pi]
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    gamma = (gamma + np.pi) % (2*np.pi) - np.pi

    return np.array([alpha, beta, gamma])


def quaternion_to_euler_zxz(q):

    q = q/np.linalg.norm(q)

    alpha = np.arctan2(
        2*(q[0]*q[3]-q[1]*q[2]),
        q[0]**2+q[2]**2-q[1]**2-q[3]**2
    )

    beta = np.arcsin((2*(q[0]*q[1]+q[2]*q[3])).clip(-1,1))

    gamma = np.arctan2(
        2*(q[0]*q[2]-q[1]*q[3]),
        q[0]**2+q[3]**2-q[1]**2-q[2]**2
    )    

    return np.array([alpha, beta, gamma])

def euler_to_quaternion_xyz(e):
    
    roll, pitch, yaw = e
    
    cr = np.cos(roll/2);
    sr = np.sin(roll/2);
    cp = np.cos(pitch/2);
    sp = np.sin(pitch/2);
    cy = np.cos(yaw/2);
    sy = np.sin(yaw/2);

    q0 = cr*cp*cy + sr*sp*sy;
    q1 = sr*cp*cy - cr*sp*sy;
    q2 = cr*sp*cy + sr*cp*sy;
    q3 = cr*cp*sy - sr*sp*cy;

    return np.array([q0, q1, q2, q3])

def euler_to_quaternion_zxz(e):

    alpha, beta, gamma = e

    ca = np.cos(alpha/2)
    sa = np.sin(alpha/2)
    cb = np.cos(beta/2)
    sb = np.sin(beta/2)
    cg = np.cos(gamma/2)
    sg = np.sin(gamma/2)

    q0 = ca*cb*cg - sa*sb*sg
    q1 = ca*sb*cg - sa*cb*sg
    q2 = ca*cb*sg + sa*sb*cg
    q3 = sa*cb*cg + ca*sb*sg

    return np.array([q0, q1, q2, q3])
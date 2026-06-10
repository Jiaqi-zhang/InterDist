# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from utils.jittor_compat import jt
import numpy as np

_EPS4 = np.finfo(np.float32).eps * 4.0

_FLOAT_EPS = np.finfo(np.float32).eps

# PyTorch-backed implementations
def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = jt.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qinv_np(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    return qinv(jt.from_numpy(q).float()).numpy()


def qnormalize(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    return q / jt.norm(q, dim=-1, keepdim=True)


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = jt.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return jt.stack((w, x, y, z), dim=1).view(original_shape)


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = jt.cross(qvec, v, dim=1)
    uuv = jt.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qeuler(q, order, epsilon=0, deg=True):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = jt.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = jt.asin(jt.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = jt.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = jt.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = jt.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = jt.asin(jt.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = jt.asin(jt.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = jt.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = jt.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = jt.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = jt.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = jt.asin(jt.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = jt.asin(jt.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = jt.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = jt.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = jt.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = jt.asin(jt.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = jt.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise

    if deg:
        return jt.stack((x, y, z), dim=1).view(original_shape) * 180 / np.pi
    else:
        return jt.stack((x, y, z), dim=1).view(original_shape)


# Numpy-backed implementations

def qmul_np(q, r):
    q = jt.from_numpy(q).contiguous().float()
    r = jt.from_numpy(r).contiguous().float()
    return qmul(q, r).numpy()


def qrot_np(q, v):
    q = jt.from_numpy(q).contiguous().float()
    v = jt.from_numpy(v).contiguous().float()
    return qrot(q, v).numpy()


def qeuler_np(q, order, epsilon=0, use_gpu=False):
    if use_gpu:
        q = jt.from_numpy(q).cuda().float()
        return qeuler(q, order, epsilon).cpu().numpy()
    else:
        q = jt.from_numpy(q).contiguous().float()
        return qeuler(q, order, epsilon).numpy()


def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def euler2quat(e, order, deg=True):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.view(-1, 3)

    ## if euler angles in degrees
    if deg:
        e = e * np.pi / 180.

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = jt.stack((jt.cos(x / 2), jt.sin(x / 2), jt.zeros_like(x), jt.zeros_like(x)), dim=1)
    ry = jt.stack((jt.cos(y / 2), jt.zeros_like(y), jt.sin(y / 2), jt.zeros_like(y)), dim=1)
    rz = jt.stack((jt.cos(z / 2), jt.zeros_like(z), jt.zeros_like(z), jt.sin(z / 2)), dim=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1

    return result.view(original_shape)


def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5 * theta).reshape(-1, 1)
    xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)


def euler_to_quaternion(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = np.stack((np.cos(x / 2), np.sin(x / 2), np.zeros_like(x), np.zeros_like(x)), axis=1)
    ry = np.stack((np.cos(y / 2), np.zeros_like(y), np.sin(y / 2), np.zeros_like(y)), axis=1)
    rz = np.stack((np.cos(z / 2), np.zeros_like(z), np.zeros_like(z), np.sin(z / 2)), axis=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1

    return result.reshape(original_shape)


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = jt.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = jt.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quaternion_to_matrix_np(quaternions):
    q = jt.from_numpy(quaternions).contiguous().float()
    return quaternion_to_matrix(q).numpy()


def quaternion_to_cont6d_np(quaternions):
    rotation_mat = quaternion_to_matrix_np(quaternions)
    cont_6d = np.concatenate([rotation_mat[..., 0], rotation_mat[..., 1]], axis=-1)
    return cont_6d


def quaternion_to_cont6d(quaternions):
    rotation_mat = quaternion_to_matrix(quaternions)
    cont_6d = jt.cat([rotation_mat[..., 0], rotation_mat[..., 1]], dim=-1)
    return cont_6d


def cont6d_to_matrix(cont6d, eps=1e-6):
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    x_raw = cont6d[..., 0:3]  
    y_raw = cont6d[..., 3:6]  
    
    x = x_raw / jt.norm(x_raw, dim=-1, keepdim=True)
    z_raw = jt.cross(x, y_raw, dim=-1)  
    parallel_mask = jt.norm(z_raw, dim=-1) < eps
    
    z = z_raw / (jt.norm(z_raw, dim=-1, keepdim=True) + 1e-12)  
    y = jt.cross(z, x, dim=-1)  
    
    x = x[..., None]
    y = y[..., None]
    z = z[..., None]
    
    mat = jt.cat([x, y, z], dim=-1)  # [..., 3, 3]  
    
    eye3 = jt.eye(3, device=cont6d.device, dtype=cont6d.dtype)
    expand_shape = list(parallel_mask.shape) + [3, 3]  
    eye_expanded = eye3.view(*([1] * parallel_mask.ndim), 3, 3).expand(*expand_shape)
    mat = jt.where(parallel_mask.unsqueeze(-1).unsqueeze(-1), eye_expanded, mat)  
    return mat  


def cont6d_to_matrix_bug(cont6d):
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]

    x = x_raw / jt.norm(x_raw, dim=-1, keepdim=True)
    z = jt.cross(x, y_raw, dim=-1)
    z = z / jt.norm(z, dim=-1, keepdim=True)

    y = jt.cross(z, x, dim=-1)

    x = x[..., None]
    y = y[..., None]
    z = z[..., None]

    mat = jt.cat([x, y, z], dim=-1)
    return mat


def cont6d_to_matrix_np(cont6d):
    q = jt.from_numpy(cont6d).contiguous().float()
    return cont6d_to_matrix(q).numpy()


def qpow(q0, t, dtype=jt.float):
    ''' q0 : tensor of quaternions
    t: tensor of powers
    '''
    q0 = qnormalize(q0)
    theta0 = jt.acos(q0[..., 0])

    ## if theta0 is close to zero, add epsilon to avoid NaNs
    mask = (theta0 <= 10e-10) * (theta0 >= -10e-10)
    theta0 = (1 - mask) * theta0 + mask * 10e-10
    v0 = q0[..., 1:] / jt.sin(theta0).view(-1, 1)

    if jt.is_tensor(t):
        q = jt.zeros(t.shape + q0.shape)
        theta = t.view(-1, 1) * theta0.view(1, -1)
    else:  ## if t is a number
        q = jt.zeros(q0.shape)
        theta = t * theta0

    q[..., 0] = jt.cos(theta)
    q[..., 1:] = v0 * jt.sin(theta).unsqueeze(-1)

    return q.to(dtype)


def qslerp(q0, q1, t):
    '''
    q0: starting quaternion
    q1: ending quaternion
    t: array of points along the way

    Returns:
    Tensor of Slerps: t.shape + q0.shape
    '''

    q0 = qnormalize(q0)
    q1 = qnormalize(q1)
    q_ = qpow(qmul(q1, qinv(q0)), t)

    return qmul(q_,
                q0.contiguous().view(jt.Size([1] * len(t.shape)) + q0.shape).expand(t.shape + q0.shape).contiguous())


def qbetween(v0, v1):
    '''
    find the quaternion used to rotate v0 to v1
    '''
    assert v0.shape[-1] == 3, 'v0 must be of the shape (*, 3)'
    assert v1.shape[-1] == 3, 'v1 must be of the shape (*, 3)'

    v = jt.cross(v0, v1, dim=-1)
    w = jt.sqrt((v0 ** 2).sum(dim=-1, keepdim=True) * (v1 ** 2).sum(dim=-1, keepdim=True)) + (v0 * v1).sum(dim=-1,
                                                                                                              keepdim=True)
    return qnormalize(jt.cat([w, v], dim=-1))


def qbetween_np(v0, v1):
    '''
    find the quaternion used to rotate v0 to v1
    '''
    assert v0.shape[-1] == 3, 'v0 must be of the shape (*, 3)'
    assert v1.shape[-1] == 3, 'v1 must be of the shape (*, 3)'

    v0 = jt.from_numpy(v0).float()
    v1 = jt.from_numpy(v1).float()
    return qbetween(v0, v1).numpy()


def lerp(p0, p1, t):
    if not jt.is_tensor(t):
        t = jt.Tensor([t])

    new_shape = t.shape + p0.shape
    new_view_t = t.shape + jt.Size([1] * len(p0.shape))
    new_view_p = jt.Size([1] * len(t.shape)) + p0.shape
    p0 = p0.view(new_view_p).expand(new_shape)
    p1 = p1.view(new_view_p).expand(new_shape)
    t = t.view(new_view_t).expand(new_shape)

    return p0 + t * (p1 - p0)

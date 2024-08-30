from sklearn.neighbors import NearestNeighbors
import numpy as np
import open3d as o3d

def multiply_T_P_homo(P2_1, T1_2):
    # P2_2_ = T1_2 * P2_1 starts
    P2_1_reshaped = P2_1.reshape(P2_1.shape[0]*P2_1.shape[1],P2_1.shape[2])
    #print(P2_1_reshaped.shape)
    
    P2_2_ = transformed_points.reshape(P2_1.shape[0],P2_1.shape[1],P2_1.shape[2])
    # print(P2_2_.shape)
    # P2_2_ = T1_2 * P2_1 ends
    return P2_2_


def nearest_neighbor(src_points: np.array, dst_points: np.array):
    '''
    Find the nearest neighbor between the point clouds
    
    Args:
      src_points(np.array): The coordinates of the first point cloud.
      dst_points(np.array): The coordiantes of the second point cloud.

    Return:
        (np.ndarray): Distances from the src_points to the closest points in dst_points
        (np.ndarray): Indices of the nearest points in the point clouds.
    '''
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst_points)
    neigh_dist, neigh_ind = neigh.kneighbors(src_points, return_distance=True)
    return neigh_dist.ravel(), neigh_ind.ravel()

def calculate_best_transform(src_points: np.array, dst_points: np.array):
    '''
    Calculates the best transform that maps  points between two point clouds.
    The cross-covariance matrix between the point clouds is calculated, then
    rotations and translations are extracted using singular value decomposition.
    
    Args:
      src_points(np.array): The coordinates of the first point cloud.
      dst_points(np.array): The coordiantes of the second point cloud.

    Returns:
        (np.array)(src_points.shape[1]+1)x(src_points.shape[1]+1) homogeneous transformation matrix that maps src_points on to dst_points
        (np.array)(src_points.shape[1]xsrc_points.shape[1]) rotation matrix
        (np.array)(src_points.shape[1]x1) translation vector
    '''
    # translate points to their centroids
    centroid_src_points = np.mean(src_points, axis=0)
    centroid_dst_points = np.mean(dst_points, axis=0)
    # compute covariance
    cov = np.cov(centroid_src_points, centroid_dst_points)
    #cov = np.mean(src_points*dst_points, axis=0) - centroid_src_points * centroid_dst_points
    #cov =  compute_cross_covariance(src_points, dst_points, centroid_src_points, centroid_dst_points)
    # rotation matrix
    U, S, Vt = np.linalg.svd(cov)
    R = np.dot(Vt.T, U.T)
    # get number of dimensions
    m = src_points.shape[1]
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1,:] *= -1
        R = np.dot(Vt.T, U.T)
    R_3 = np.eye(3)
    R_3[:2,:2]=R
    # translation
    t = centroid_dst_points.T - np.dot(R_3,centroid_src_points.T)
    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R_3
    T[:m, m] = t
    return T, R, t

def icp(src_points: np.array, dst_points: np.array, init_pose: np.array=None, max_iterations: int=20, tolerance: float=0.001):
    '''
    Finds best-fit transform that maps points in src_points on to points in dst_points
    using Iterative Closest Point method.

    Input:
        src_points(np.array): The coordinates of the first point cloud.
        dst_points(np.array): The coordiantes of the second point cloud.
        init_pose(np.array):  homogeneous transformation
        max_iterations(int): exit algorithm after max_iterations
        tolerance(float): convergence criteria
   
    Return:
        (np.array)(src_points.shape[1]+1)x(src_points.shape[1]+1) homogeneous transformation matrix that maps src_points on to dst_points
        Euclidean distances (errors) of the nearest neighbor
        number of iterations to converge
    '''
    # get number of dimensions
    m = src_points.shape[1]
    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1, src_points.shape[0]))
    dst = np.ones((m+1, dst_points.shape[0]))
    src[:m,:] = np.copy(src_points.T)
    dst[:m,:] = np.copy(dst_points.T)
    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)
    prev_error = 0
    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)
        # compute the transformation between the current source and nearest destination points
        T,_,_ = calculate_best_transform(src[:m,:].T, dst[:m,indices].T)
        # update the current source
        src = np.dot(T, src)
        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    # calculate final transformation
    T,_,_ = calculate_best_transform(src_points, src[:m,:].T)
    return T, distances, i

if __name__ == '__main__':
    working_directory = '/Users/tunaseckin/Desktop/'
    recovered_path = 'dust3r/Recovered_Pose_3DPoints/'
    Pts3D = np.load(working_directory + recovered_path+f"P1.npy") # read recovered PointCloud
    PoseT = np.load(working_directory + recovered_path+f"T1.npy")

    Pts3D_shaped_1 = Pts3D.reshape(Pts3D.shape[0]*Pts3D.shape[1],Pts3D.shape[2]) # reshape it as 3D point list.

    Pts3D_2 = np.load(working_directory + recovered_path+f"P2.npy") # read recovered PointCloud
    PoseT = np.load(working_directory + recovered_path+f"T2.npy")


    Pts3D_shaped_2 = Pts3D_2.reshape(Pts3D_2.shape[0]*Pts3D_2.shape[1],Pts3D_2.shape[2]) # reshape it as 3D point list.

    T, distances, i = icp(Pts3D_shaped_2, Pts3D_shaped_1, np.linalg.inv(PoseT))
    print(T)

    ones = np.ones((Pts3D_2.shape[0]*Pts3D_2.shape[1],1))
    # make homogeneous by adding 1 to 3D point
    P2_1_homo = np.hstack((Pts3D_shaped_2,ones))
    # Apply the transformation to our point cloud
    transformed_points = np.dot(T, P2_1_homo.T).T
    Pts3D_shaped_2_T = transformed_points[:,:-1] # convert back to cartesian coord


    pcd1 = o3d.geometry.PointCloud() # initialize point cloud
    pcd1.points = o3d.utility.Vector3dVector(Pts3D_shaped_1) # set points of the point cloud

    pcd_list = []
    pcd_list.append(pcd1)

    pcd2 = o3d.geometry.PointCloud() # initialize point cloud
    pcd2.points = o3d.utility.Vector3dVector(Pts3D_shaped_2_T) # set points of the point cloud
    pcd_list.append(pcd2)

    o3d.visualization.draw_geometries(pcd_list, window_name=f"Point Cloud of 2 images") # display the point cloud.




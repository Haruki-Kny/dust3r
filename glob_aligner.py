import sys
import scipy
import trimesh
import numpy as np
import numpy.ma as ma
import os
import torch
from scipy.spatial.transform import Rotation
from datetime import datetime
import pickle
# import rerun as rr

sys.path.append("..")
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes

import pdb # for debugging

def transformation_matrix(sx, sy, rotation_matrix, tx, ty):
    # Scale matrix
    scale_matrix = np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Rotation matrix extended to 4x4
    rotation_matrix_extended = np.array([
        [rotation_matrix[0, 0], rotation_matrix[0, 1], 0, 0],
        [rotation_matrix[1, 0], rotation_matrix[1, 1], 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Translation matrix
    translation_matrix = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Combine transformations: T * R * S
    transformation_matrix = translation_matrix + rotation_matrix_extended @ scale_matrix

    return transformation_matrix

def convert3x3to4x4(trans3x3):
    trans4x4 = np.eye(4)
    trans4x4[:2,:2] = trans3x3[:2,:2]
    trans4x4[:2,3] = trans3x3[:2,2]
    return trans4x4

def generate_unique_filename(prefix="", extension="txt"):
    # Get the current date and time, and format it
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the filename
    if prefix == "":
        filename = f"{current_time}.{extension}"
    else:
        filename = f"{prefix}_{current_time}.{extension}"

    return filename

def get_3D_model_from_scene_list(outdir, silent, scene_list, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05):
    """
    extract 3D_model (glb file) from a list of reconstructed scenes
    """
    scene = trimesh.Scene()
    for scene_idx, scene_item in enumerate(scene_list):
        if scene_item is None:
            continue
        # # post processes
        # if clean_depth:
        #     scene_item = scene_item.clean_pointcloud()
        # if mask_sky:
        #     scene_item = scene_item.mask_sky()

        # scene_list.append((imgs, focals, poses, pts3d_list, confidence_masks))
        # get optimized values from scene_item
        rgbimg = scene_item[0]
        focals = scene_item[1].cpu()
        cams2world = scene_item[2].cpu()
        # 3D pointcloud from depthmap, poses and intrinsics
        pts3d = to_numpy(scene_item[3])
        # scene_item.min_conf_thr = float(scene_item.conf_trf(torch.tensor(min_conf_thr)))
        msk = to_numpy(scene_item[4])

        # full pointcloud
        if as_pointcloud:
            # all images in one scene
            # pts = np.concatenate([p[m] for p, m in zip(pts3d, msk)])
            # col = np.concatenate([p[m] for p, m in zip(rgbimg, msk)])
            # single image in one scene
            pts = pts3d[0][msk[0]]
            col = rgbimg[0][msk[0]]
            pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
            scene.add_geometry(pct)
        else:
            meshes = []
            # all images in one scene
            # for i in range(len(rgbimg)):
                # meshes.append(pts3d_to_trimesh(rgbimg[i], pts3d[i], msk[i]))
                # show all points, mask -> False
                # meshes.append(pts3d_to_trimesh(rgbimg[i], pts3d[i], np.ones_like(msk[i])))
            # single image in one scene
            meshes.append(pts3d_to_trimesh(rgbimg[0], pts3d[0], msk[0]))
            mesh = trimesh.Trimesh(**cat_meshes(meshes))
            scene.add_geometry(mesh)

    outfile = os.path.join(outdir, generate_unique_filename(prefix="scene", extension="glb"))
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile

# Inputs:
# - P, a (n,dim) [or (dim,n)] matrix, a point cloud of n points in dim dimension.
# - Q, a (n,dim) [or (dim,n)] matrix, a point cloud of n points in dim dimension.
# P and Q must be of the same shape.
# This function returns :
# - Pt, the P point cloud, transformed to fit to Q
# - (T,t) the affine transform
def affine_registration(P, Q):
    transposed = False
    if P.shape[0] < P.shape[1]:
        transposed = True
        P = P.T
        Q = Q.T
    (n, dim) = P.shape
    # Compute least squares
    p, res, rnk, s = scipy.linalg.lstsq(np.hstack((P, np.ones([n, 1]))), Q)
    # Get translation
    t = p[-1].T
    # Get transform matrix
    T = p[:-1].T
    # Compute transformed pointcloud
    Pt = P@T.T + t
    if transposed: Pt = Pt.T
    return Pt, (T, t)

# Rigidly (+scale) aligns two point clouds with know point-to-point correspondences
# with least-squares error.
# Returns (scale factor c, rotation matrix R, translation vector t) such that
#   Q = P*cR + t
# if they align perfectly, or such that
#   SUM over point i ( | P_i*cR + t - Q_i |^2 )
# is minimised if they don't align perfectly.
def umeyama(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    return c, R, t


if __name__ == '__main__':
    device = 'cpu'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 100

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    # path = '../images/06/image_2/'
    path = '/Users/tunaseckin/Desktop/TUM/TUM Practical Courses/Practical Courses/Geometric Scene Understanding/DUST3R/pragueparks/KITTI(color)/sequences(color)/06/image_2/'
    image_filename_ls = [str(i).zfill(6) + '.png' for i in range(600, 900, 1)]
    # image_filename_ls = ['000000.png', '000001.png', '000002.png', '000003.png', '000004.png', '000005.png']
    image_list = [path + image_filename for image_filename in image_filename_ls]
    scene_list = []
    scale_factor_cur2prev = 1.0
    prev_pose = np.eye(4)
    prev_pts3d = None
    prev_trf_cur2nex = np.eye(4)
    prev_confidence_masks = None
    # transformation matrix (original, first iamge in first window -> next)
    trf_org2cur = None

    for idx in range(len(image_list)-1):
        pts3d_list = []
        print(image_list[idx],image_list[idx+1])

        images = load_images([image_list[idx],image_list[idx+1]], size=512)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=batch_size)

        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
        #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
        # in each view you have:
        # an integer image identifier: view1['idx'] and view2['idx']
        # the img: view1['img'] and view2['img']
        # the image shape: view1['true_shape'] and view2['true_shape']
        # an instance string output by the dataloader: view1['instance'] and view2['instance']
        # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
        # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
        # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

        # next we'll use the global_aligner to align the predictions
        # depending on your task, you may be fine with the raw output and not need it
        # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
        # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

        # retrieve useful values from scene:
        imgs = scene.imgs
        # imgs = scene.imgs[0], scene.imgs[0]
        # pdb.set_trace()
        # imgs[0][...,0]=0
        # imgs[0][...,1]=1
        # imgs[0][...,2]=0
        # imgs[1][...,0]=1
        # imgs[1][...,1]=1
        # imgs[1][...,2]=0
        focals = scene.get_focals()
        poses = scene.get_im_poses()
        pts3d = scene.get_pts3d()
        confidence_masks = scene.get_masks()
        # pdb.set_trace()

        # visualize reconstruction
        #scene.show()
        print("len(pose):", len(poses))
        print("poses[0]:\n", poses[0])
        print("poses[1]:\n", poses[1])
        print("len(pts3d):", len(pts3d))
        print("focals:\n", focals)

        # * pose[1] is the transformation matrix (current->next)
        trf_cur2nex = np.dot(poses[1].detach().numpy(), np.linalg.inv(poses[0].detach().numpy()))

        current_pts3d = pts3d[0].detach().numpy()
        print("cur_pts3d shape", current_pts3d.shape)
        scaling_affine = np.ones((1,3))
        scaling_icp = np.ones((1,3))

        # * find 2D-2D matches between the two images
        from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
        pts2d_list, pts3d_list = [], []
        for i in range(2):
            conf_i = confidence_masks[i].cpu().numpy()
            pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
            pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
        reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
        print(f'found {num_matches} matches')
        matches_im1 = pts2d_list[1][reciprocal_in_P2]
        matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]
        #scale_of_matches, _, _ = umeyama(matches_im0, matches_im1)
        pts3d_list = []

        cu = 1
        scale_of_matches = 1
        new_transformation = np.eye(4)
        if idx > 0:
            # make prev_pts3d to the same coordinate system as current_pts3d
            print("shape of prev_trf_cur2nex:", prev_trf_cur2nex.shape)
            print("shape of prev_pts3d:", prev_pts3d.shape)

            # Transform from shape (144, 512, 3) to (73728, 3)
            prev_points = prev_pts3d.reshape(-1, 3)
            # Transform from shape (73728, 3) to (73728, 4) and add a column of ones at the end
            points_homogeneous = np.hstack((prev_points, np.ones((prev_points.shape[0], 1))))
            # Apply the transformation matrix
            transformed_prev_pts3d = np.dot(prev_trf_cur2nex, points_homogeneous.T).T
            # Transform from shape (73728, 4) to (73728, 3)
            transformed_prev_pts3d = transformed_prev_pts3d[:, :3].reshape(prev_pts3d.shape)
            # print("transformed_prev_pts3d shape", transformed_prev_pts3d.shape)
            # print("current_pts3d shape", current_pts3d.shape)
            # pts3d_list.append(transformed_prev_pts3d)
            # scale_factor_cur2prev = np.median(transformed_prev_pts3d[:, :, 2]) / np.median(current_pts3d[:, :, 2])

            # Apply mask to transformed_prev_pts3d and current_pts3d
            both_masks = np.logical_and(prev_confidence_masks, confidence_masks[0])
            # pdb.set_trace()
            # * Calculate the scale factor by median of z values
            # scale_factor_cur2prev = np.median(transformed_prev_pts3d[both_masks == 1, 2]) / np.median(current_pts3d[both_masks == 1, 2])
            # pdb.set_trace()
            # * by mean
            scale_factor_cur2prev = np.mean(transformed_prev_pts3d[both_masks == 1, 2]) / np.mean(current_pts3d[both_masks == 1, 2])
            # find scaling using  closed-form least square method:affine registration
            transformed_prev_pts3d_rspd = transformed_prev_pts3d.reshape(transformed_prev_pts3d.shape[0]*transformed_prev_pts3d.shape[1],transformed_prev_pts3d.shape[2])
            current_pts3d_rspd = current_pts3d.reshape(current_pts3d.shape[0]*current_pts3d.shape[1],current_pts3d.shape[2])
            Pt, (Tra, tra) = affine_registration(transformed_prev_pts3d_rspd, current_pts3d_rspd)
            scaling_affine[0,0]=(Tra[0,0]**2 + Tra[1,0]**2 + Tra[2,0]**2)**(1/2)
            scaling_affine[0,1]=(Tra[0,1]**2 + Tra[1,1]**2 + Tra[2,1]**2)**(1/2)
            scaling_affine[0,2]=(Tra[0,2]**2 + Tra[1,2]**2 + Tra[2,2]**2)**(1/2)
            print("scale_factor_affine_registration:",scaling_affine)

            # find scaling using ICP
            from ICP_align import icp
            transformed_prev_pts3d_rspd = transformed_prev_pts3d.reshape(transformed_prev_pts3d.shape[0]*transformed_prev_pts3d.shape[1],transformed_prev_pts3d.shape[2])
            current_pts3d_rspd = current_pts3d.reshape(current_pts3d.shape[0]*current_pts3d.shape[1],current_pts3d.shape[2])
            icp_T, _, _ = icp(transformed_prev_pts3d_rspd,current_pts3d_rspd, poses[0].detach().numpy())       
            scaling_icp[0,0]=(icp_T[0,0]**2 + icp_T[1,0]**2 + icp_T[2,0]**2)**(1/2)
            scaling_icp[0,1]=(icp_T[0,1]**2 + icp_T[1,1]**2 + icp_T[2,1]**2)**(1/2)
            scaling_icp[0,2]=(icp_T[0,2]**2 + icp_T[1,2]**2 + icp_T[2,2]**2)**(1/2)
            print("scale_factor_ICP:",scaling_icp)
            scale_factor_cur2prev = np.mean(scaling_icp)
            #find scaling using umeyama
            #cu, Ru, tu = umeyama(transformed_prev_pts3d_rspd, current_pts3d_rspd)
            #print("scale_factor_umeyama point cloud:", cu)
            scale_of_matches, Ru, tu = umeyama(matches_im0, matches_im1)
            print("scale_factor_umeyama matched points", scale_of_matches)
            ident = np.eye(4)
            
            new_transformation= ident #convert3x3to4x4(Tra) #transformation_matrix(scale_of_matches, scale_of_matches, Ru, tu[0], tu[1])
            print("scale_factor_cur2prev:", scale_factor_cur2prev)
            # for debugging
            tmp_sclaed_cur_pts3d = scale_factor_cur2prev * current_pts3d
            points_homogeneous = np.hstack((tmp_sclaed_cur_pts3d.reshape(-1, 3), np.ones((tmp_sclaed_cur_pts3d.reshape(-1, 3).shape[0], 1))))
            transformed_cur_pts3d = np.dot(poses[0].detach().numpy(), points_homogeneous.T).T
            transformed_cur_pts3d = transformed_cur_pts3d[:, :3].reshape(current_pts3d.shape)
            # pts3d_list.append(transformed_cur_pts3d)

        # * apply scaling on the transformation matrix
        scaled_trf_cur2nex = trf_cur2nex.copy()
        # various scaling optimizations. APPLY ONLY ONE OF THEM
        #scale_factor_cur2prev = cu # apply umeyama point clouds result of scaling
        #scale_factor_cur2prev = scale_of_matches # apply umeyama matched points result of scaling
        scaled_trf_cur2nex[:3, 3] = scale_factor_cur2prev * trf_cur2nex[:3, 3] #
        # apply affine registration result for scaling
        #scaled_trf_cur2nex[0, 3] = scaling_affine[0,0] * trf_cur2nex[0, 3]
        #scaled_trf_cur2nex[1, 3] = scaling_affine[0,1] * trf_cur2nex[1, 3]
        #scaled_trf_cur2nex[2, 3] = scaling_affine[0,2] * trf_cur2nex[2, 3]
        # apply ICP result for scaling
        #scaled_trf_cur2nex[0, 3] = scaling_icp[0,0] * trf_cur2nex[0, 3]
        #scaled_trf_cur2nex[1, 3] = scaling_icp[0,1] * trf_cur2nex[1, 3]
        #scaled_trf_cur2nex[2, 3] = scaling_icp[0,2] * trf_cur2nex[2, 3]
        # apply new transformation as a whole, not only scaled translation
        #scaled_trf_cur2nex = new_transformation
        print("scaled_trf_cur2nex:\n", scaled_trf_cur2nex)

        if idx == 0: # org is the first coordinate system
            trf_org2cur = poses[0].detach().numpy()
        elif idx > 0:
            # * calculate the transformation matrix (original, first iamge in first window -> current)
            trf_org2cur = prev_trf_cur2nex @ trf_org2cur
        print("trf_org2cur:\n", trf_org2cur)

        # * transform point cloud (pts3d)
        for i in range(len(poses)):
            if idx == 0:
                # pts3d_list.append(np.zeros(pts3d[i].shape)) # for debugging
                pts3d_list.append(pts3d[i].detach().numpy())
            else:
            # transform point cloud to the original coordinate system

            # if (idx, i) in [(0, 0), (1, 1)]: # for debugging
            #     pts3d_list.append(np.zeros(pts3d[i].shape))
            # else:
                # print("shown (idx, i) = ", (idx, i))
                # the below is the original code
                tmp_pts3d = pts3d[i].detach().numpy().copy().reshape(-1, 3)
                tmp_pts3d *= scale_factor_cur2prev
                tmp_pts3d = np.hstack((tmp_pts3d, np.ones((tmp_pts3d.shape[0], 1))))
                # transformed_pts3d = np.dot(np.linalg.inv(trf_org2cur), tmp_pts3d.T).T # failure (why?)
                transformed_pts3d = np.dot(trf_org2cur, tmp_pts3d.T).T # success
                # pdb.set_trace()
                transformed_pts3d = transformed_pts3d[:, :3].reshape(current_pts3d.shape)
                pts3d_list.append(transformed_pts3d)
            # if (idx, i) == (0, 1): # green for the first window, for debugging
            #     print("shown (idx, i) = ", (idx, i))
            #     # 1st window, 2nd image
            #     imgs[1][...,0]=0
            #     imgs[1][...,1]=1
            #     imgs[1][...,2]=0

            # print(f"original_pts3d[{i}]: {pts3d[i]}" )
            # print(f"transformed_pts3d[{i}]: {transformed_pts3d}" )

        # * save the scene in the original coordinate system
        scene_list.append((imgs, focals, poses, pts3d_list, confidence_masks))

        prev_trf_cur2nex = scaled_trf_cur2nex
        prev_pts3d = pts3d[1].detach().numpy()
        prev_confidence_masks = confidence_masks[1].detach().numpy()

                # * find 2D-2D matches between the two images
        from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
        pts2d_list, pts3d_list = [], []
        for i in range(2):
            conf_i = confidence_masks[i].cpu().numpy()
            pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
            pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
        reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
        print(f'found {num_matches} matches')
        matches_im1 = pts2d_list[1][reciprocal_in_P2]
        matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

        # visualize a few matches
        import numpy as np
        from matplotlib import pyplot as plt
        n_viz = 10
        match_idx_to_viz = np.round(np.linspace(0, num_matches-1, n_viz)).astype(int)
        viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

        H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
        img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img = np.concatenate((img0, img1), axis=1)
        plt.figure()
        figure_title = image_list[idx][image_list[idx].rfind("/")+1:] + "-" + image_list[idx+1][image_list[idx+1].rfind("/")+1:]
        print(figure_title)
        plt.title(figure_title)
        #plt.imshow(img)
        cmap = plt.get_cmap('jet')
        for i in range(n_viz):
            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
            plt.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
        # plt.savefig('../output/' + figure_title)  # Save the figure before showing it
        #plt.show(block=True)

    # visualize the final scene in mesh -> disabled
    # outdir = '../output'
    # as_pointcloud = False
    # outfile_path = get_3D_model_from_scene_list(outdir=outdir, silent=False, scene_list=scene_list, min_conf_thr=3, as_pointcloud=as_pointcloud,
    #                         mask_sky=False, clean_depth=False, transparent_cams=False, cam_size=0.05)

    # * save the scene_list in pickle format
    with open(f'/Users/tunaseckin/Desktop/dust3r/output/scene_list_data/' + generate_unique_filename('', 'pkl'), 'wb') as f:
        pickle.dump(scene_list, f)

    # * Rerun: if you run on a local machine, you can visualize with rerun
    # rr.init("rerun_example_my_data", spawn=True)
    # rr.log_file_from_path(outfile_path)

    print("final pose:")
    # print(adjusted_pose)
    print(len(scene_list))
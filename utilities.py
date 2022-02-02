import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from pyntcloud import PyntCloud
from glob import glob
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
import open3d as o3d
import os
from datetime import datetime
import re
import scipy
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from tqdm.notebook import tqdm, trange
from multiprocessing import Pool

db_path = r'../icip2020_perry_quality_repack_bin'
waterloo_path = r'../src/the_WPC_database'
deg_metrics_path = os.path.join('data', 'icip2020_deg_metrics.json')
degraded_pcs_features_path = os.path.join('data', 'icip2020_degraded_pcs_features.csv')
degraded_pcs_features_preds_path = os.path.join('data', 'icip2020_degraded_pcs_features_preds.csv')
block_bits = 6
block_shape = [2**block_bits] * 3
bbox_min = [0,0,0]

class pc :
    #def __init__(self): 
    def load_tree(self): ### Quach
        if self.is_ref == True :
            self.tree = cKDTree(self.points[['x','y','z']])
        else :
            self.tree = cKDTree(self.points[['x','y','z']], balanced_tree=False)
    def load_dists_ngbs(self):  ### Quach
        if self.is_ref == False :
            self.dists_AB, self.ngb_AB = self.tree.query(self.ref.points[['x','y','z']], n_jobs=-1)
            self.dists_BA, self.ngb_BA = self.ref.tree.query(self.points[['x','y','z']], n_jobs=-1)  
    def compute_normals(self, radius = 100, max_nn = 30):
        np_points = self.points[['x','y','z']].to_numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius = radius, max_nn = max_nn))
        ndf = pd.DataFrame(np.asarray(pcd.normals), columns=['nx', 'ny','nz'])
        self.points = pd.concat([self.points, ndf], axis = 1)
        
    def downsample(self, voxel_size=10, downsampled_size= 2048):
        np_points=self.points[['x','y','z']].to_numpy()
        pcd = o3d.geometry.PointCloud()    
        pcd.points = o3d.utility.Vector3dVector(np_points)
        downpcd = pcd.voxel_down_sample(voxel_size = voxel_size)
        _list=[]
        for i in random.sample(range(len(np.asarray(downpcd.points))), downsampled_size):
            _list.append(np.asarray(downpcd.points)[i])
        self.downsampled = np.asarray(_list)
        self.downsampled_size = downsampled_size
        
    def partition(self):   ### Quach
        geo_bits = self.geometry_bits
        octree_bits = geo_bits - block_bits
        ref_pc_xyz = self.points[['x','y','z']]
        bbox_max = np.asarray([2**geo_bits] * 3)
        ref_pc_xyz = ref_pc_xyz[np.all(ref_pc_xyz < bbox_max, axis=1)]
        blocks_meta = partition_octree(ref_pc_xyz, bbox_min, bbox_max, octree_bits)['blocks_meta']
        self.bbox_max = bbox_max
        self.octree_bits = octree_bits
        self.block_bits = block_bits
        self.blocks_meta = blocks_meta
        self.num_blocks = len(blocks_meta.keys())
        
    def compute_features(self):  ### Quach
        if self.is_ref == False:
            ref_pc = self.ref.points
            deg_pc = self.points
            ngb_AB = self.ngb_AB
            ngb_BA = self.ngb_BA
            ref_pc_ngb = deg_pc.iloc[ngb_AB]
            deg_pc_ngb = ref_pc.iloc[ngb_BA]

            xyz = ['x', 'y', 'z']
            nxyz = ['nx', 'ny', 'nz']
            ref_pc_xyz = ref_pc[xyz].values
            ref_pc_xyz_ngb = ref_pc_ngb[xyz].values
            ref_pc_nxyz_ngb = ref_pc_ngb[nxyz].values
            deg_pc_xyz = deg_pc[xyz].values
            deg_pc_xyz_ngb = deg_pc_ngb[xyz].values
            deg_pc_nxyz_ngb = deg_pc_ngb[nxyz].values

            max_energy = 3 * ((2 ** self.geometry_bits) ** 2)

            features = {}
            features['input_points'] = len(ref_pc)
            features['output_points'] = len(deg_pc)
            features['out_in_points_ratio'] = len(ref_pc) / len(deg_pc)

            features['d1_mse_AB'] = np.mean(np.square(self.dists_AB))
            features['d1_mse_BA'] = np.mean(np.square(self.dists_BA))
            features['max_d1_mse'] = max(features['d1_mse_AB'], features['d1_mse_BA'])

            features['d1_psnr_AB'] = psnr(features['d1_mse_AB'], max_energy)
            features['d1_psnr_BA'] = psnr(features['d1_mse_BA'], max_energy)
            features['min_d1_psnr'] = min(features['d1_psnr_AB'], features['d1_psnr_BA'])

            features['max_energy'] = max_energy

            ref_pc_xyz_diff = ref_pc_xyz - ref_pc_xyz_ngb
            deg_pc_xyz_diff = deg_pc_xyz - deg_pc_xyz_ngb

            features['d2_mse_AB'] = np.mean(np.sum(ref_pc_xyz_diff * ref_pc_nxyz_ngb, axis=1) ** 2, axis=0)
            features['d2_mse_BA'] = np.mean(np.sum(deg_pc_xyz_diff * deg_pc_nxyz_ngb, axis=1) ** 2, axis=0)
            features['max_d2_mse'] = max(features['d2_mse_AB'], features['d2_mse_BA'])

            features['d2_psnr_AB'] = psnr(features['d2_mse_AB'], max_energy)
            features['d2_psnr_BA'] = psnr(features['d2_mse_BA'], max_energy)
            features['min_d2_psnr'] = min(features['d2_psnr_AB'], features['d2_psnr_BA'])
            
            self.features = features
            
    def find_shared_blocks(self):
        if self.is_ref == False :
            shared_list = []
            for block in self.blocks_meta.keys():
                if block in self.ref.blocks_meta.keys():
                    shared_list.append(block)
            self.shared_blocks = shared_list
                    
            
        
class icip_pc(pc):
    def __init__(self, pc_name, codec_id, codec_rate, geometry_bits, mos, mos_ci, relative_path, radius):
        self.id = (pc_name, codec_id, codec_rate)
        self.pc_name = pc_name
        self.codec_id = codec_id
        self.codec_rate = codec_rate
        self.geometry_bits = geometry_bits
        self.mos = mos
        self.mos_ci = mos_ci
        self.relative_path = relative_path
        self.radius = radius
        if self.codec_id == 'REF':
            self.is_ref = True
        else:
            self.is_ref = False
        self.std =0
        self.spec = 'icip'
        self.sum_var = 0
        self.predicted_mos = 0
    
    def load_points(self):   ### Quach
        plydata = PlyData.read(os.path.join(db_path, self.relative_path))
        pc_df = pd.DataFrame(plydata['vertex'].data).astype(np.float32)
        new_cols = {x: re.sub('scalar_', '', re.sub(r'_\(\d+\)', '', x)).lower() for x in list(pc_df)}
        pc_df = pc_df.rename(columns=new_cols)
        self.points = pc_df
        self.size = pc_df.shape[0]
        
    def connect_with_ref(self,pcs):
        if self.is_ref == False :
            for pc in pcs:
                if pc.pc_name == self.pc_name and pc.is_ref :
                    self.ref = pc
        
            
class wpc_pc(pc):
    def __init__(self, content, pc_name, distortion, mos, is_ref):
        self.pc_name = pc_name
        self.content = content
        self.geometry_bits = 10
        self.mos = mos
        self.std =0
        self.spec = 'waterloo'
        self.is_ref = is_ref
        self.distortion = distortion
        self.blocks = {}
        self.id = self.content
            
    def load_points(self):
        if self.is_ref:
            original_pcs_path = os.path.join(waterloo_path, 'original_PCs')
            plydata = PlyData.read(os.path.join(original_pcs_path, self.content+'.ply'))
            pc_df = pd.DataFrame(plydata['vertex'].data).astype(np.float32)
            self.points = pc_df
            self.size = pc_df.shape[0]
        else:
            distorted_pcs_path = os.path.join(waterloo_path, 'distorted_PCs')
            for entry in os.listdir(distorted_pcs_path):
                if self.pc_name in entry:
                    self.pc_name = entry[:-4]
                    plydata = PlyData.read(os.path.join(distorted_pcs_path, entry))
                    pc_df = pd.DataFrame(plydata['vertex'].data).astype(np.float32)
                    self.points = pc_df
                    self.size = pc_df.shape[0]
            
    def connect_with_ref(self,pcs):
        if self.is_ref == False :
            for pc in pcs:
                if pc.content == self.content and pc.is_ref :
                    self.ref = pc 
                
    def write_to_disk(self, path, textfile):
        for block in self.blocks_meta.keys() : 
            np_points = self.blocks_meta[block]['block']
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_points)
            if self.is_ref:
                o3d.io.write_point_cloud(os.path.join(path, '__' + self.pc_name + '__'+ str(block[0])+ '__'+ str(block[1])+ '__'+ str(block[2]) + '.ply'), pcd)
                textfile.write(os.path.join(path, '__' + self.pc_name + '__'+ str(block[0])+ '__'+ str(block[1])+ '__'+ str(block[2]) + '.ply' + "\n"))
            else :
                o3d.io.write_point_cloud(os.path.join(path, '__' + self.pc_name + '__'+ str(block[0])+ '__'+ str(block[1])+ '__'+ str(block[2]) + '.ply'), pcd)
                textfile.write(os.path.join(path, '__' + self.pc_name + '__'+ str(block[0])+ '__'+ str(block[1])+ '__'+ str(block[2]) + '.ply' + "\n"))
            
    def delete_blocks_meta (self):
        del self.blocks_meta
       
    def compute_number_of_blocks(self):
        self.num_blocks = len(self.blocks.keys())
            

        
class modelnet_pc(pc):
    def __init__(self, pc_name, is_ref = True):
        self.std = 0
        self.spec = 'modelnet'
        self.is_ref = is_ref
        self.geometry_bits = 6
        if self.is_ref :
            self.pc_name = pc_name.split('/')[-1][:-4]
            self.content = pc_name.split('/')[-3]
    def load_points(self, path):
        ptc = PyntCloud.from_file(path)
        ret = ptc.points[['x','y','z']]
        self.points = ret
        self.size = ret.shape[0]
    def write_to_disk(self, path, d2):
        np_points = self.points[['x','y','z']].to_numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_points)
        o3d.io.write_point_cloud(os.path.join(path, '__' + self.content + '__' +self.pc_name+ '__'+ str(self.std)+ '__' + str(d2) + '.ply'), pcd)

def make_noisy_version(pc , std):
    if pc.spec == 'icip':
    #return pc['pc'][['x','y','z']].applymap(lambda x: x + np.random.normal(0,var,pc.shape))
        new_pc = icip_pc(pc.pc_name, pc.codec_id, pc.codec_rate, pc.geometry_bits, None , None, None , pc.radius)
        new_pc.points = np.round_(np.clip(pc.points[['x','y','z']]+np.random.normal(0, std, pc.points[['x','y','z']].shape), 0, 1023))
        new_pc.std = std
        new_pc.ref = pc
        return new_pc
    if pc.spec == 'modelnet':
        new_pc = modelnet_pc(pc.pc_name, False)
        new_pc.points = np.round_(np.clip(pc.points[['x','y','z']]+np.random.normal(0, std, pc.points[['x','y','z']].shape), 0, 63))
        new_pc.std = std
        new_pc.ref = pc
        new_pc.pc_name = pc.pc_name
        new_pc.content = pc.content
        return new_pc
    else :
        print('error')
            

##### The rest of the code is taken from Maurice Quach Github project


def esc_latex(s):
    return re.sub('_', '\\_', s)
def pts_to_vx(pts, block_shape, block):
    block.fill(0)
    if pts is None or len(pts) == 0:
        return block
    pts = pts.astype(np.uint32)
    block[pts[:, 0], pts[:, 1], pts[:, 2]] = 1
    return block

def psnr(value, max_energy):
    if value == 0:
        return 75
    return 10 * np.log10(max_energy / value)

#@njit
def mse_features_njit(ref_pc_val, ref_pc_val_ngb, deg_pc_val, deg_pc_val_ngb):
    ref_pc_val_mask = ~np.isnan(ref_pc_val) & ~np.isinf(ref_pc_val) & ~np.isnan(ref_pc_val_ngb) & ~np.isinf(ref_pc_val_ngb)
    ref_pc_val = ref_pc_val[ref_pc_val_mask]
    ref_pc_val_ngb = ref_pc_val_ngb[ref_pc_val_mask]

    deg_pc_val_mask = (~np.isnan(deg_pc_val)) & (~np.isinf(deg_pc_val)) & ~np.isnan(deg_pc_val_ngb) & ~np.isinf(deg_pc_val_ngb)
    deg_pc_val = deg_pc_val[deg_pc_val_mask]
    deg_pc_val_ngb = deg_pc_val_ngb[deg_pc_val_mask]
    
    f_AB = np.mean(np.square(ref_pc_val - ref_pc_val_ngb))
    f_BA = np.mean(np.square(deg_pc_val - deg_pc_val_ngb))
    f_max = max(f_AB, f_BA)
    
    return f_AB, f_BA, f_max
    
def mse_features(col_name, features, ref_pc, ref_pc_ngb, deg_pc, deg_pc_ngb):
    key_AB = f'{col_name}_mse_AB'
    key_BA = f'{col_name}_mse_BA'
    key_max = f'max_{col_name}_mse'
    result = mse_features_njit(ref_pc[col_name].values, ref_pc_ngb[col_name].values,
                               deg_pc[col_name].values, deg_pc_ngb[col_name].values)
    features[key_AB], features[key_BA], features[key_max] = result
    
    
    
def compute_new_bbox(idx, bbox_min, bbox_max):
    midpoint = (bbox_max - bbox_min) // 2 + bbox_min
    # Compute global block bounding box
    cur_bbox_min = bbox_min.copy()
    cur_bbox_max = midpoint.copy()
    if idx & 1:
        cur_bbox_min[0] = midpoint[0]
        cur_bbox_max[0] = bbox_max[0]
    if (idx >> 1) & 1:
        cur_bbox_min[1] = midpoint[1]
        cur_bbox_max[1] = bbox_max[1]
    if (idx >> 2) & 1:
        cur_bbox_min[2] = midpoint[2]
        cur_bbox_max[2] = bbox_max[2]

    return cur_bbox_min, cur_bbox_max


# Partitions points using an octree scheme
# returns points in local coordinates (block) and octree structure as an 8 bit integer (right to left order)
def split_octree(points, bbox_min, bbox_max):
    ret_points = [[] for x in range(8)]
    midpoint = (bbox_max - bbox_min) // 2
    global_bboxes = [compute_new_bbox(i, bbox_min, bbox_max) for i in range(8)]
    # Translate into local block coordinates
    # Use local block bounding box
    local_bboxes = [(np.zeros(3), x[1] - x[0]) for x in global_bboxes]
    for point in points:
        location = 0
        if point[0] >= midpoint[0]:
            location |= 0b001
        if point[1] >= midpoint[1]:
            location |= 0b010
        if point[2] >= midpoint[2]:
            location |= 0b100
        ret_points[location].append(point - np.pad(global_bboxes[location][0], [0, len(point) - 3]))
    binstr = 0b00000000
    for i, rp in enumerate(ret_points):
        if len(rp) > 0:
            binstr |= (0b00000001 << i)

    return [np.vstack(rp) for rp in ret_points if len(rp) > 0], binstr, local_bboxes


# Returns list of blocks and octree structure as a list of 8 bit integers
# Recursive octree partitioning function that is slow for high number of points (> 500k)
def partition_octree_rec(points, bbox_min, bbox_max, level):
    if len(points) == 0:
        return [points], None
    if level == 0:
        return [points], None
    bbox_min = np.asarray(bbox_min)
    bbox_max = np.asarray(bbox_max)
    ret_points, binstr, bboxes = split_octree(points, bbox_min, bbox_max)
    result = [partition_octree_rec(rp, bbox[0], bbox[1], level - 1) for rp, bbox in zip(ret_points, bboxes)]
    blocks = [subblock for block_res in result for subblock in block_res[0] if len(subblock) > 0]
    new_binstr = [binstr] + [subbinstr for block_res in result if block_res[1] is not None for subbinstr in block_res[1]]
    return blocks, new_binstr

def group_by_block(points_local, block_idx, block_len):
    blocks = [np.zeros((l, points_local.shape[1])) for l in block_len]
    blocks_last_idx = np.zeros(len(block_len), dtype=np.uint32)
    for i, b_idx in enumerate(block_idx):
        blocks[b_idx][blocks_last_idx[b_idx]] = points_local[i]
        blocks_last_idx[b_idx] += 1
    return blocks

def partition_octree(points, bbox_min, bbox_max, level):
    points = np.asarray(points)
    if len(points) == 0:
        return [points], None
    if level == 0:
        return [points], None
    bbox_min = np.asarray(bbox_min)
    np.testing.assert_array_equal(bbox_min, [0, 0, 0])
    bbox_max = np.asarray(bbox_max)
    out_of_bounds = points[~np.all(points < bbox_max, axis=1)]
    assert len(out_of_bounds) == 0, out_of_bounds
    geo_level = int(np.ceil(np.log2(np.max(bbox_max))))
    block_level = geo_level - level
    assert geo_level >= level
    block_size = 2 ** block_level

    # Compute partitions for each point
    block_ids = points[:, :3] // block_size
    block_ids = block_ids.astype(np.uint32)
    # np.unique is the slowest part of this function
    # may be worth it to investigate a better implementation for this
    block_ids_unique, block_idx, block_len = np.unique(block_ids, return_inverse=True, return_counts=True, axis=0)

    # Interleave coordinate bits to reorder by octree block order
    sort_key = []
    for x, y, z in block_ids_unique:
        zip_params = [f'{v:0{level}b}' for v in [z, y, x]]
        #assert len(set([len(s) for s in zip_params])) == 1, 
        f'zip_params: {zip_params}, zyx: {z}, {y}, {x}, level: {level}, geo_level: {geo_level}, block_level: {block_level}'
        sort_key.append(''.join(i + j + k for i, j, k in zip(*zip_params)))
    sort_idx = np.argsort(sort_key)
    block_ids_unique = block_ids_unique[sort_idx]
    block_len = block_len[sort_idx]
    # perform reordering by inverting permutation
    inv_sort_idx = np.zeros_like(sort_idx)
    inv_sort_idx[sort_idx] = np.arange(sort_idx.size)
    # Compute new block idx for each point
    block_idx = inv_sort_idx[block_idx]

    # Translate points into local block coordinates
    local_refs = np.pad(block_ids_unique[block_idx] * block_size, [[0, 0], [0, points.shape[1] - 3]])
    points_local = points - local_refs

    # Group points by block
    blocks = group_by_block(points_local, block_idx, block_len)

    # Build binary string recursively using the block_ids
    _, binstr = partition_octree_rec(block_ids_unique, [0, 0, 0], (2 ** level) * np.array([1, 1, 1]), level)

    blocks_meta = {}
    for i in range(len(block_ids_unique)):
        block_id = block_ids_unique[i]
        block_data = {'block_id': tuple(block_id), 'local_ref': block_id * block_size, 'block_size': block_size, 'block': blocks[i]}
        blocks_meta[tuple(block_id)] = block_data
    
    return {
        'blocks': blocks,
        'binstr': binstr,
        'blocks_meta': blocks_meta,
    }

def df_to_pc(df):
    points = df[['x','y','z']].values
    return points


def pa_to_df(points):
    cols = ['x', 'y', 'z', 'red', 'green', 'blue']
    types = (['float32'] * 3) + (['uint8'] * 3)
    d = {}
    assert 3 <= points.shape[1] <= 6
    for i in range(points.shape[1]):
        col = cols[i]
        dtype = types[i]
        d[col] = points[:, i].astype(dtype)
    df = pd.DataFrame(data=d)
    return df


def pc_to_df(pc):
    points = pc.points
    return pa_to_df(points)


def load_pc(path):
    logger.debug(f"Loading PC {path}")
    pc = PyntCloud.from_file(path)
    ret = df_to_pc(pc.points)
    logger.debug(f"Loaded PC {path}")

    return ret


def write_pc(path, pc):
    df = pc_to_df(pc)
    write_df(path, df)


def write_df(path, df):
    pc = PyntCloud(df)
    pc.to_file(path)


def get_shape_data(resolution, data_format):
    assert data_format in ['channels_last', 'channels_first']
    bbox_min = 0
    bbox_max = resolution
    p_max = np.array([bbox_max, bbox_max, bbox_max])
    p_min = np.array([bbox_min, bbox_min, bbox_min])
    if data_format == 'channels_last':
        dense_tensor_shape = np.concatenate([p_max, [1]]).astype('int64')
    else:
        dense_tensor_shape = np.concatenate([[1], p_max]).astype('int64')

    return p_min, p_max, dense_tensor_shape


def get_files(input_glob):
    return np.array(glob(input_glob, recursive=True))


def load_points(files, batch_size=32):
    files_len = len(files)

    with multiprocessing.Pool() as p:
        logger.info('Loading PCs into memory (parallel reading)')
        points = np.array(list(tqdm(p.imap(load_pc, files, batch_size), total=files_len)))

    return points

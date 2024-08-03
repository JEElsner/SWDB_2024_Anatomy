import pandas as pd
import numpy as np
import cloudvolume
from cloudfiles import CloudFiles
from meshparty import skeleton
import skeleton_plot as skelplot
import warnings
warnings.filterwarnings('ignore')

def load_em_skeleton_as_meshwork(skeleton_id):
    # skeleton_id: the root id of one skeleton
    input_directory = "precomputed://gs://allen_neuroglancer_ccf/em_minnie65_v1078"
    cv_obj = cloudvolume.CloudVolume(input_directory, use_https = True) # Initialize cloud volume
    cv_sk = cv_obj.skeleton.get(skeleton_id) #load an example skeleton
    
    sk = skeleton.Skeleton(cv_sk.vertices, 
                       cv_sk.edges, 
                       vertex_properties={'radius': cv_sk.radius,
                                          'compartment': cv_sk.compartment}, 
                       root = len(cv_sk.edges), # the final edge is root
                       remove_zero_length_edges = False)

    conversion_factor = 1000
    
    return sk, conversion_factor

def load_em_skeleton_as_df(skeleton_id):
    # skeleton_id: the root id of one skeleton
    input_directory = "precomputed://gs://allen_neuroglancer_ccf/em_minnie65_v1078"
    cv_obj = cloudvolume.CloudVolume(input_directory, use_https = True) # Initialize cloud volume
    cv_sk = cv_obj.skeleton.get(skeleton_id) #load an example skeleton
    
    sk = skeleton.Skeleton(cv_sk.vertices, 
                       cv_sk.edges, 
                       vertex_properties={'radius': cv_sk.radius,
                                          'compartment': cv_sk.compartment}, 
                       root = len(cv_sk.edges), # the final edge is root
                       remove_zero_length_edges = False)

    conversion_factor = 1000
    
    skel_df = pd.DataFrame({'vertex_xyz': [x for x in cv_sk.vertices],
                        'vertex_x': [x[0] for x in cv_sk.vertices],
                        'vertex_y': [x[1] for x in cv_sk.vertices],
                        'vertex_z': [x[2] for x in cv_sk.vertices],
                        'd_path_um': sk.distance_to_root / conversion_factor,
                        'compartment': cv_sk.compartment, 
                        'presyn_counts': cv_sk.presyn_counts, 
                        'presyn_size': cv_sk.presyn_size, 
                        'postsyn_counts': cv_sk.postsyn_counts, 
                        'postsyn_size': cv_sk.postsyn_size,})
    skel_df.index.names = ['vertex_index']
    
    return skel_df

def load_em_segmentprops_to_df():
    input_directory = "precomputed://gs://allen_neuroglancer_ccf/em_minnie65_v1078"
    cv_obj = cloudvolume.CloudVolume(input_directory, use_https = True) # Initialize cloud volume
    
    cf = CloudFiles(cv_obj.cloudpath)
    
    # get segment info
    segment_properties = cf.get_json("segment_properties/info")

    segment_tag_values = np.array(segment_properties['inline']['properties'][1]['values'])

    segment_tags = np.array(segment_properties['inline']['properties'][1]['tags'])
    segment_tags_map = pd.Series(np.array(segment_properties['inline']['properties'][1]['tags']))
    segment_tags_map = segment_tags_map.to_dict()

    # map values to root id
    seg_df = pd.DataFrame({
        'nucleus_id': segment_properties['inline']['properties'][0]['values'],
        segment_properties['inline']['properties'][2]['id']: segment_properties['inline']['properties'][2]['values'],
        segment_properties['inline']['properties'][3]['id']: segment_properties['inline']['properties'][3]['values'],
        segment_properties['inline']['properties'][4]['id']: segment_properties['inline']['properties'][4]['values'], 
        segment_properties['inline']['properties'][5]['id']: segment_properties['inline']['properties'][5]['values'],
        'cell_type': segment_tag_values[:,0],
        'brain_area': segment_tag_values[:,1],
    },
        index=segment_properties['inline']['ids'])

    # map tags to root id
    seg_df['cell_type'] = seg_df.cell_type.replace(segment_tags_map)
    seg_df['brain_area'] = seg_df.brain_area.replace(segment_tags_map)

    return seg_df

def load_lm_skeleton_as_meshwork(skeleton_id):
    # skeleton_id: the root id of one skeleton
    input_directory = "precomputed://s3://aind-open-data/exaSPIM_609281_2022-11-03_13-49-18_reconstructions/precomputed"
    cv_obj = cloudvolume.CloudVolume(input_directory) # Initialize cloud volume
    cv_sk = cv_obj.skeleton.get(skeleton_id) #load an example skeleton
    
    sk = skeleton.Skeleton(cv_sk.vertices, 
                           cv_sk.edges, 
                           vertex_properties={'radius': cv_sk.radius,
                                              'compartment': cv_sk.compartment,
                                              'allenId': cv_sk.allenId}, 
                           root = 0, 
                           # root = len(sk_em.edges), # when the final edge is root
                           remove_zero_length_edges = False)

    conversion_factor = 1 #for LM (data in microns )
    
    return sk, conversion_factor
"""
Helper routines for extracting ccf information from skeletons and ccf
properties such as the "name" or acronym" for a given ccf id.

"""
from copy import deepcopy
from numpy import int64

import math
import numpy as np
import pandas as pd


CCF_ATLAS = pd.read_csv('/data/adult_mouse_ccf_structures.csv')
CCF_PROPERTY = [
    "acronym",
    "color_hex_triplet",
    "graph_order",
    "hemisphere_id",
    "id",
    "name",
    "parent_structure_id",
    "structure_id_path",
]


def get_ccf_property(ccf_id, ccf_property, depth=None, print_id=False):
    """
    Gets the value of the property for a given ccf id. For example, suppose
    that ccf_id=1000 and ccf_property="name", then this routine returns the
    name of the brain region corresponding to the ccf id.

    Parameters
    ----------
    ccf_id : int
        Numerical ID of a ccf region.
    ccf_property : str
        Property of a brain region that is stored in the ccf. The global
        variable "CCF_PROPERTIES" lists all of the properties.

    Returns
    -------
    str/float/int
        Value of the property for a given ccf id.

    """
    # Check that inputs are valid
    err_msg = ccf_property + " is not ccf property!"
    assert ccf_property in CCF_PROPERTY, err_msg

    # Return property value
    if depth is not None:
        ccf_id = get_ccf_id_by_depth(ccf_id, depth)

    if ccf_id in CCF_ATLAS["id"]:
        return CCF_ATLAS.loc[CCF_ATLAS["id"] == ccf_id, ccf_property].iloc[0]
    else:
        return "NaN"        


def get_ccf_id_by_depth(ccf_id, depth):
    if ccf_id in CCF_ATLAS["id"]:
        ccf_id_hierarchy = get_ccf_property(ccf_id, "structure_id_path")
        ccf_id_hierarchy = list(map(int, ccf_id_hierarchy.split("/")[1:-1]))
        return ccf_id_hierarchy[min(depth, len(ccf_id_hierarchy) - 1)]
    else:
        return ccf_id

    
def get_ccf_ids(skel, compartment_type=None, vertex_type=None, depth=None):
    # Get ccf ids from "CCF_ATLAS"
    skel = deepcopy(skel)
    vertices = get_vertices(skel, compartment_type, vertex_type)
    ccf_ids = skel.vertex_properties["ccf"][vertices]

    # Return ccf ids
    if depth is not None:
        return [get_ccf_id_by_depth(ccf_id, depth) for ccf_id in ccf_ids]
    else:
        return ccf_ids


def is_valid(ccf_id):
    """
    Determines whether "ccf_id" is valid, meaning that it is not 'NaN' and it
    is an contained in "CCF_ATLAS["id"]".

    Parameters
    ----------
    ccf_id : int
        ccf id to be checked.

    Returns
    -------
    bool
        Indication of whether "ccf_id" is valid.

    """
    is_nan = np.isnan(ccf_id)
    in_dataframe = True if any(CCF_ATLAS["id"] == ccf_id) else False
    return False if is_nan or not in_dataframe else True


def get_vertices(skel, compartment_type, vertex_type):
    # Special Cases
    if compartment_type == 1:
        return [skel.root]
    elif not compartment_type and not vertex_type:
        return skel.vertex_properties['compartment'] != -1

    # General Cases
    if compartment_type and not vertex_type:
        return skel.vertex_properties['compartment'] == compartment_type
    else:
        assert vertex_type in ["branch_points", "end_points"]
        verts = skel.end_points if vertex_type == "end_points" else skel.branch_points
        if compartment_type:
            vertex_compartments = skel.vertex_properties['compartment']
            return [v for v in verts if vertex_compartments[v] == compartment_type]
        else:
            return verts


def get_connectivity_matrix(skels, compartment_type, binary=False, depth=None):
    # Initializations
    ccf_ids_list = list()
    for skel in skels:
        ccf_ids_list.append(
            get_ccf_ids(
                skel,
                compartment_type=compartment_type,
                depth=depth,
                vertex_type="end_points",
            )
        )
    ccf_ids, cnts = np.unique(ccf_ids_list, return_counts=True)
    ccf_ids = ccf_ids[cnts > 5]
    cnts = cnts[cnts > 5]

    region_to_idx = dict({r: idx for idx, r in enumerate(regions[cnts > 10])})

    # Populate matrix
    matrix = np.zeros((len(skels), len(region_to_idx)))
    for i, ccf_ids in enumerate(ccf_ids_list):
        ccf_ids, cnts = np.unique(ccf_ids, return_counts=True)
        for j, ccf_id in enumerate(ccf_ids):
            if not math.isnan(ccf_id) and ccf_id in region_to_idx.keys():
                matrix[i, region_to_idx[ccf_id]] = cnts[j]
    idx_to_region = {idx: r for r, idx in region_to_idx.items()}
    return (matrix > 0 if binary else matrix), idx_to_region

from cloudvolume import CloudVolume
from meshparty import skeleton

import boto3

BUCKET = "aind-open-data"
DATASET_KEYS = [
    "exaSPIM_609281_2022-11-03_13-49-18_reconstructions",
    "exaSPIM_651324_2023-03-06_15-13-25_reconstructions",
    "exaSPIM_653158_2023-06-01_20-41-38_reconstructions",
    "exaSPIM_653980_2023-08-10_20-08-29_reconstructions",
    "mouselight_reconstructions",
]


def number_of_samples():
    """
    Returns the number of samples in the light microscopy dataset.

    Parameters
    ----------
    None

    Returns
    -------
    int
        Number of samples in the light microscopy dataset.

    """
    return len(DATASET_KEYS)

def load_lm_datasets():
    """
    Loads all of the light microscopy neurons across four exaspim datasets and
    and the mouse light dataset.

    Parameters
    ----------
    None

    Returns
    -------
    list[meshparty.skeleton.Skeleton]
        Skeletons that represent light microscopy neurons.

    """    
    skeletons = list()
    for key in DATASET_KEYS:
        prefix = f"{key}/precomputed/"
        cv_dataset = CloudVolume(f"precomputed://s3://{BUCKET}/{prefix}")
        skeletons.extend(load_skeletons(cv_dataset, prefix + "skeleton/"))    
    return skeletons


def load_skeletons(cv_dataset, prefix):
    """
    Loads all skeletons from a cloudvolume dataset.

    Parameters
    ----------
    cv_dataset : CloudVolume
        Dataset that contains a set of skeletons.
    prefix : str
        Directory located in S3 bucket that contains skeletons to be loaded.
        This value is used to get the object ids of the skeletons.

    Returns
    -------
    list[meshparty.skeleton.Skeleton]
        Skeletons from cloudvolume dataset.

    """
    skeletons = list()
    for skel_id in get_skeleton_ids(prefix):
        skeletons.append(get_skeleton(cv_dataset, skel_id))
    return skeletons


def get_skeleton_ids(prefix):
    """
    Extracts skeleton ids from the corresponding S3 path that points to a
    given skeleton.

    Parameters
    ----------
    skeleton_paths : list[str]
        Paths to skeletons that are stored in an S3 bucket.

    Returns
    -------
    list[int]
        Skeleton ids extracted from "skeleton_paths".

    """
    skeleton_ids = list()
    for file in list_files_in_prefix(prefix):
        filename = file.split("/")[-1]
        if filename.isnumeric():
            skeleton_ids.append(int(filename))
    return skeleton_ids


def list_files_in_prefix(prefix):
    """
    Lists all files that are stored in the directory "prefix" which is located
    in an S3 bucket.

    Parameters
    ----------
    prefix : str
        Directory located in S3 bucket to be read from.

    Returns
    -------
    list[str]
        Files stored in the directory "prefix" which located in an S3 bucket.

    """
    # Initializations
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')

    # Main
    files = list()
    pages = paginator.paginate(Bucket=BUCKET, Prefix=prefix, Delimiter='/')
    for page in pages:
        for obj in page.get('Contents', []):
            files.append(obj['Key'])
    return files


def get_skeleton(cv_dataset, skel_id):
    cv_skel = cv_dataset.skeleton.get(skel_id)
    skel = skeleton.Skeleton(
        cv_skel.vertices, 
        cv_skel.edges,
        remove_zero_length_edges=False,
        root=0,
        vertex_properties=set_vertex_properties(cv_skel),
    )
    return skel


def set_vertex_properties(cv_skel):
    """
    Sets the vertex properties of mesh party skeleton.

    Parameters
    ----------
    cv_skel : CloudVolume.skeleton
    """
    vertex_properties = {
        "ccf": cv_skel.allenId,
        "radius": cv_skel.radius,
        "compartment": cv_skel.compartment,
    }
    return vertex_properties

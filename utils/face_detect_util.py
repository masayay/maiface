import torch
import os, glob
from datetime import datetime
import pathlib, shutil
from PIL import Image
import redis
import conf
"""
Load Configuration
"""
EMBEDDINGS = conf.EMBEDDINGS
REDIS_HOST = conf.REDIS_HOST
REDIS_PORT = conf.REDIS_PORT
REDIS_PASS = conf.REDIS_PASS
FAISS_IDX = conf.FAISS_IDX
DETECT_IDX = conf.DETECT_IDX

class UtilException(Exception):
    def __init__(self, message):
        super().__init__(message)

def connect_redis(h=REDIS_HOST, p=REDIS_PORT, d=FAISS_IDX, pw=REDIS_PASS):
    pool = redis.ConnectionPool(host=h, port=p, db=d, password=pw)
    r = redis.StrictRedis(connection_pool=pool)
    
    return r

# for writing results on redis db
def connect_redis2(h=REDIS_HOST, p=REDIS_PORT, d=DETECT_IDX, pw=REDIS_PASS):
    pool = redis.ConnectionPool(host=h, port=p, db=d, password=pw)
    r = redis.StrictRedis(connection_pool=pool)
    
    return r

def get_torch_device():
    """
    Determine if an nvidia GPU is available
    
    Returns
    -------
    device:
        torch.device('cuda:0')
        torch.device('cpu')
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    return device

def get_faceid(file_path):
    """
    Get last directory as faceid from file_path
    
    Parameters
    ----------
    file_path : str
       Example) '/data/embedding_root/faceid/XXX.npy'
       Example) '/data/picture_root/faceid/XXX.jpg'
    
    Returns
    -------
    faceid : str
    
    """
    if not os.path.exists(file_path):
        raise UtilException("ERROR: %s is not exsists." % file_path)
        
    dirname = os.path.dirname(file_path)
    faceid = os.path.basename(dirname)

    return faceid

def get_newest_npy_file_date(faceid_path):
    """
    Get newest numpy file creation date and number of numpy files under faceid_path

    Parameters
    ----------
    faceid_path : str
        Example) '/data/embedding_root/faceid'

    Returns
    -------
    num_numpy_file: int
    newest_timestamp : datetime.datetime
        If file is not exsists, return 2020-01-01
    """
    file_name = os.path.join(faceid_path, "*.npy")
    # Glab all npy files under comapny id
    file_list = glob.glob(file_name)
    if len(file_list) > 0:
        # Sort by file creation date old to new
        file_list.sort(key = os.path.getctime, reverse=True)
        # Get timestamp
        newest_timestamp = datetime.fromtimestamp(os.stat(file_list[0]).st_ctime)
        
    else:
        newest_timestamp = datetime(2020,1,1)
    
    return len(file_list), newest_timestamp

def resize_image_by_width(image, dst_width):
    """
    Resize image with fix aspect ratio
    
    Parameters
    ----------
    image : PIL.Image
    dst_width: int
       resize target width
    
    Returns
    -------
    dst : PIL.Image
    """
    if not isinstance(image, Image.Image):
        raise UtilException("ERROR: %s is not PIL image." % image )
    
    dst_width = int(dst_width)
    org_width, org_hight = image.size
    dst_height = round(org_hight * (dst_width / org_width))
    dst_image = image.resize((dst_width, dst_height))

    return dst_image
    
def get_num_faceid():
    """
    Get number of faceid including *.npy under EMBEDDINGS

    Returns
    -------
    int
    """
    num_faceid = 0
    try:
        for path in pathlib.Path(EMBEDDINGS).iterdir():
            if path.is_dir():
                file_name = os.path.join(path, "*.npy")
                file_list = glob.glob(file_name)
                if len(file_list) > 0:
                    num_faceid += 1
    except FileNotFoundError:
        pass
    
    return num_faceid

def get_num_npy_files():
    """
    Get number of *npy files under EMBEDDINGS


    Returns
    -------
    {'faceid1': 8, 'faceid2':2 ...}

    """
    num_npy_files = {}
    try:
        for path in pathlib.Path(EMBEDDINGS).iterdir():
            if path.is_dir():
                file_name = os.path.join(path, "*.npy")
                file_list = glob.glob(file_name)
                num_embedding = len(file_list)
                if num_embedding > 0:
                    faceid = os.path.basename(path)
                    num_npy_files[faceid] = num_embedding
    except FileNotFoundError:
        pass
    
    return num_npy_files

def get_num_npy_files_in_faceid(faceid):
    file_name = os.path.join(EMBEDDINGS, faceid, "*.npy")
    file_list = glob.glob(file_name)
    if len(file_list) > 0:
        return faceid, len(file_list)
    else:
        return None, 0

def delete_faceid_files(faceid_path):
    """
    Delete faceid and all numpy files

    Parameters
    ----------
    class_path : str
        Example) '/data/embedding_root/faceid'

    Returns
    0 : not delete
    1 : delete files
    -------
    """
    result = 0
    if os.path.isdir(faceid_path):
        try:
            shutil.rmtree(faceid_path)
            result = 1
        except FileNotFoundError:
            pass

    return result

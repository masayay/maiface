#!/usr/bin/python3
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torch
import os, glob, time, sys
import numpy as np
from logging import getLogger, FileHandler, Formatter
from utils import face_detect_util as fdu
import conf
"""
Load Configuration
"""
GPU = conf.ENABLE_GPU
EMBEDDINGS_BATCH_OUT = conf.EMBEDDINGS_BATCH_OUT
PICTURES = conf.PICTURES
CACHE_DIR = conf.CACHE_DIR
WORKERS = conf.WORKERS
BATCH_MODE = conf.BATCH_MODE
PIC_WIDTH = conf.PIC_WIDTH
PIC_HIGHT = conf.PIC_HIGHT
"""
device
"""
# Get available device
if GPU:
    device = fdu.get_torch_device()
else:
    device = 'cpu'
"""
mtcnn
"""
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20, device=device
    )
"""
resnetv1
"""
# Set cache directory for pretrained model
os.environ['TORCH_HOME'] = CACHE_DIR
# Load model
resnetv1 = InceptionResnetV1(pretrained='vggface2').to(device).eval()
"""
Logging Configuration
"""
logger = getLogger(__name__)
logger.setLevel(conf.BATCH_LOG_LEVEL)
if not logger.hasHandlers():
    handler = FileHandler(filename=conf.BATCH_LOGFILE) 
    handler.setFormatter(Formatter("%(asctime)s %(levelname)6s %(message)s"))
    logger.addHandler(handler)

def main():
    # Time stamp
    t1 = time.time()
    logger.info("START DETECTION. DEVICE USE: %s" % device)
    if BATCH_MODE:
        logger.info('Start batch mode')
        feature_extraction_batch(PICTURES, EMBEDDINGS_BATCH_OUT)
    else:
        logger.info('Start normal mode')
        feature_extraction(PICTURES, EMBEDDINGS_BATCH_OUT)
        
    # Time stamp
    t2 = time.time()
    # Finish
    logger.info('Finish! Elapse time: %s sec' % int(t2 - t1))
    sys.exit()


def collate_pil(batch):
    """
    Config DataLoader output
    
    Parameters
    ----------
    batch : int

    Returns
    -------
    images : list[PIL.Image]
    targets : list[str]
    """
    images, targets = [], [] 
    for image, target in batch: 
        images.append(image) 
        targets.append(target) 
    return images, targets

def feature_extraction_batch(image_dir, npy_dir):
    """
    Create embeddings from JPEG image.
    All .jpg files will be loaded including subdirectory
    
    Parameters
    ----------
    image_path : Str
    npy_dir : Str

    Returns
    -------
    None.

    """
    # load images with PIC_HIGHT and PIC_WIDTH
    # if thre's other picutre size, it'll be transformed.
    # Due to proceed via batch, all pictures must be same size.
    batch_transform = transforms.Compose(
        [transforms.Resize(PIC_WIDTH), 
         transforms.CenterCrop((PIC_HIGHT, PIC_WIDTH))])
    dataset = datasets.ImageFolder(image_dir, transform=batch_transform)
    
    # Set output file directory 
    dataset.samples = [
        (img_path, img_path.replace(image_dir, npy_dir))
            for img_path, class_idx in dataset.samples
    ]
    
    # Define batch parameters
    batch_size = 32
    # Windows 'nt', else will be 'posix'
    workers = 0 if os.name == 'nt' else WORKERS
    loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        # Linking image and target
        collate_fn=collate_pil
    )
    
    # Crop face and extract feature vector
    for batch, (images, paths) in enumerate(loader):
        # Crop face
        img_cropped_list = mtcnn(images)
        # Check Face detected. If not rotate and try dtection again.
        detected_list = []
        for img_cropped, image, path in zip(img_cropped_list, images, paths):
            if img_cropped is not None:
                detected_list.append(img_cropped)
            else:
                img_cropped = mtcnn(image.rotate(-90))
                logger.info("Lotate picture %s" % path)
                if img_cropped is not None:
                    detected_list.append(img_cropped)
                else:
                    logger.error("Cannot detect face %s" % path)
        
        # Stack to list tensors to dim
        face_tensors = torch.stack(detected_list).to(device)
        embeddings = resnetv1(face_tensors)
        for embedding, path in zip(embeddings, paths):
            np_embedding = embedding.squeeze().to('cpu').detach().numpy().copy()
            np_path = os.path.join(
                os.path.dirname(path),
                os.path.splitext(os.path.basename(path))[0])
            os.makedirs(os.path.dirname(path) + "/", exist_ok=True)
            np.save(np_path, np_embedding)

        logger.info('Batch {} of {}'.format(batch + 1, len(loader)))


def feature_extraction(image_dir, npy_dir):
    """
    Create embeddings from JPEG image.
    All .jpg files will be loaded including subdirectory
    When fail batch mode, proceed linearly.
    
    Parameters
    ----------
    image_path : Str
    npy_dir : Str

    Returns
    -------
    None.

    """
    image_path = image_dir + "/**/*.[jJ][pP][gG]"
    # Glab all image files
    files = glob.glob(image_path, recursive=True)
    
    # Init counter
    cnt=0
    num_files = len(files)
                
    # Crop face and extract feature vector
    for file in files:    
        # Preogress
        cnt+=1
        if cnt % 10 == 0 : logger.info('{}/{}'.format(cnt,num_files))
        # Load file
        image = Image.open(file)
        # Resize image
        # bigger image take time to create embedding so change smaller
        if image.size[1] > 1024:
            image = fdu.resize_image_by_width(image, 1024)
        # Crop face
        img_cropped = mtcnn(image)
        # When cannot detect face, continue next
        if img_cropped is None:
            # Try rotate and detection
            img_cropped = mtcnn(image.rotate(-90))
            logger.info("Lotate picture %s" % file)
            if img_cropped is None:
                logger.error("Cannot detect face on %s" % file)
                continue
        # Exract feature vector
        embedding = resnetv1(img_cropped.to(device).unsqueeze(0))
        # Change tensor to ndarray & back to cpu
        np_embedding = embedding.squeeze().to('cpu').detach().numpy().copy()
        # Create output dir
        # npy_dir + faceid + file
        np_path= os.path.join(
            npy_dir,
            fdu.get_faceid(file),
            os.path.splitext(os.path.basename(file))[0])
        # Create dir if not exsist
        os.makedirs(os.path.dirname(np_path) + "/", exist_ok=True)
        # Save npy file
        np.save(np_path, np_embedding)

if __name__ == '__main__':
    main()


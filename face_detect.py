from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import faiss
import numpy as np
import glob, time, os
from datetime import datetime, timedelta
from utils import face_detect_util as fdu
import conf
from logging import getLogger
from pathlib import Path
import threading
import collections
from utils import face_api_response as far
"""
Load Configuration
"""
logger = getLogger(conf.LOG_OUTPUT)
SPEED_TEST = conf.SPEED_TEST
ENABLE_GPU = conf.ENABLE_GPU
EMBEDDINGS = conf.EMBEDDINGS
CACHE_DIR = conf.CACHE_DIR
I_DISTANCE = conf.IDENTICAL_DISTANCE
S_DISTANCE = conf.SAVE_DISTANCE
S_INTERVAL = conf.SAVE_INTERVAL
S_MINIMUM = conf.SAVE_MINIMUM
IVFFLAT = conf.IVFFLAT
FAISS_QUEUE = conf.FAISS_QUEUE
FAISS_EVENT_MANAGER = conf.FAISS_EVENT_MANAGER
ENABLE_DETECT_DB = conf.ENABLE_DETECT_DB
DETECT_EXPIRE = conf.DETECT_EXPIRE

class FaceDetector(object):
    """
    Face detector class
    """    
    def __init__(self):
        # Config device
        if ENABLE_GPU:
            self.device = fdu.get_torch_device()
            # Auto select of network algorithm
            torch.backends.cudnn.benchmark = True
        else:
            self.device = 'cpu'
        
        logger.info("FaceDetector started on device: %s" % self.device)
        # Grad calculation off
        torch.set_grad_enabled(False)
        
        """
        MTCNN
        """
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            keep_all=True, device=self.device
            )

        """
        InceptionResnetV1
        """
        # Set cache directory for pretrained model
        os.environ['TORCH_HOME'] = CACHE_DIR
        # Load model
        self.resnetv1 = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
        """
        Faiss
        """
        # Initialize faiss index
        self.index, self.index_ids = self.init_faiss()
        # Start faiss event manager
        self.faiss_event_manager("init")
        """
        Recognition output on redis
        """
        if ENABLE_DETECT_DB:
            self.auth_db = fdu.connect_redis2()
            self.auth_db.flushdb()
            logger.info("Enable detection results output.")
            

    def init_faiss(self):
        """
        Initialize Faiss
        """
        ndarray_list = [] # list of embeddings
        index_ids = [] # list of id for embeddings
        
        # load all embeding files
        npy_files = os.path.join(EMBEDDINGS, "*/*.npy") 
        files = glob.glob(npy_files)
        for file in files:
            ndarray_list.append(np.load(file))
            index_ids.append(fdu.get_faceid(file))        
        # Change list to ndarray
        ndarrays = np.array(ndarray_list, dtype='f')
        # set dimension
        dim = 512
        # Prepare IndexIVFFlat or IndexFlatL2
        num_faceid = fdu.get_num_faceid()
        if IVFFLAT:
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, num_faceid, faiss.METRIC_L2)
        else:
            index = faiss.IndexFlatL2(dim)
        # CPU or GPU
        if str(self.device).startswith('cuda:'):
            gpu_no = int(str(self.device).split(':')[1])
            res = faiss.StandardGpuResources()  # use a single GPU
            res.noTempMemory() # Avoid using GPU temp memory
            index = faiss.index_cpu_to_gpu(res, gpu_no, index)
        # train index if IVFFLAT
        if IVFFLAT:
            assert not index.is_trained
            index.train(ndarrays)
            assert index.is_trained
        # Create index
        index.add(ndarrays)
        logger.info("Faiss index initialized. People: %s Faces: %s" % (num_faceid, len(index_ids)))
        
        return index, index_ids

    def face_search(self, frame, deviceid=None):
        """
        Crop face from boxes, embedding and search comanpy id 

        Parameters
        ----------
        frame : numpy.ndarray, PIL.Image
        boxes : [[x1,y1,x2,y2],...]

        Returns
        -------
        results : Dictionary
            [{'box':[x1,y1,x2,y2], 'id':XXXX, 'probability':XXXX}, ...]
            When distance is over IDENTICAL_DISTANCE, return id as empny.
        """
        # List for result
        results = []
        
        # Releases all unoccupied cached memory on cuda
        if str(self.device).startswith('cuda:'):
            torch.cuda.empty_cache()

        if SPEED_TEST: t1 = time.time()
        
        # Detect Face box
        boxes, probs = self.mtcnn.detect(frame, landmarks=False)
            
        if SPEED_TEST: t2 = time.time()
        
        if boxes is None:
            return results
            
        # Crop Face
        img_cropped = self.mtcnn.extract(frame, boxes, None)
            
        # Get embeddings
        embeddings = self.resnetv1(img_cropped.to(self.device))
            
        if SPEED_TEST: t3 = time.time()
        if SPEED_TEST:
            logger.info('Face detection: {}msec'.format(int((t2 - t1)*1000)))
            logger.info('Face embedding: {}msec'.format(int((t3 - t2)*1000)))
            
        for box, embedding in zip(boxes, embeddings):
            # store result
            result ={}
            result["box"] = box
            # tensor to ndarray
            np_embedding = embedding.squeeze().to('cpu').detach().numpy().copy()
            # add dimension for search (vs array of embeddings )
            np_exp_embedding = np.expand_dims(np_embedding, axis=0)

            if SPEED_TEST: t4 = time.time()
            # Faiss Nearest neighbor search
            distances, neighbors = self.index.search(np_exp_embedding, 2)

            if SPEED_TEST: t5 = time.time()
            if SPEED_TEST: logger.info('Face recognition: {}msec'.format(int((t5 - t4)*1000)))

            # index nearest distance
            distance = distances.squeeze()[0].astype(float)
            if distance >= 1:
                # Distance over 1, identification probability is 0%
                result["probability"] = 0
            else:
                # Distance under 1, identification probability will be 0.-99.9%
                result["probability"] = round((1 - distance),3) *100
                
            # Label name from idx
            faceid = self.index_ids[neighbors.squeeze()[0]]
            # Add companyid into result
            # negative distance is too differ.
            if 0 <= distance < I_DISTANCE:
                result["id"] = faceid
                # Add detection result on redis db
                if ENABLE_DETECT_DB:
                    key = deviceid + "_" + faceid
                    value = str(round(distance,6))
                    self.auth_db.rpush(key, value)
                    # set and renew expire for key
                    self.auth_db.expire(key, DETECT_EXPIRE)
                    
            else:
                result["id"] = ""
                
            # Check Detection 
            logger.debug("Face detected. ID: {}, Distance: {}".format(faceid, round(distance,6)))
                
            # Save embedding
            if 0 <= distance < S_DISTANCE and result["id"] is not None:
                # Check latest embedding file date
                faceid_path = os.path.join(EMBEDDINGS, faceid)
                file_num, file_date = fdu.get_newest_npy_file_date(faceid_path)
                # If latest file past more than save interval, add
                now = datetime.now()
                if now - file_date > timedelta(days = S_INTERVAL) or file_num < S_MINIMUM:
                    self.save_embedding(result["id"], np_embedding)
                    
            # Add result
            results.append(result)
            
        # regurn results                
        return results

    def add_embedding(self, npy_file):
        """
        Parameters
        ----------
        npy_file : str
            example) '/data/embedding_root/faceid/XXX.npy'
        """
        # Add embedding to Faiss index
        np_exp_embedding = np.expand_dims(np.load(npy_file), axis=0)
        self.index.add(np_exp_embedding)
        
        # Add id of embedding
        faceid = fdu.get_faceid(npy_file)
        self.index_ids.append(faceid)
        logger.info("Embedding added: {}".format(npy_file))

    def save_embedding(self, faceid, np_embedding):
        """
        Parameters
        ----------
        faceid : str
        embedding : numpy.ndarray
        """
        # embedding save path
        faceid_path = os.path.join(EMBEDDINGS, faceid)
        if not os.path.exists(faceid_path):
            logger.info("Create new directory: %s" % faceid_path)
            os.makedirs(faceid_path, exist_ok=True)
        
        # np.save automaticaly add ".npy" on file name
        now = datetime.now()
        npy_name = os.path.join(faceid_path, faceid + "_" + now.strftime('%Y%m%d_%H%M%S') + ".npy")
        np.save(npy_name, np_embedding)
        logger.info("Embedding saved: %s" % npy_name)
        
        # Add embdding to Faiss index
        self.faiss_event_manager("add", npy_name=npy_name)

    def register_faceid(self, faceid, frame):
        """
        Parameters
        ----------
        faceid : str
        frame : PIL

        Returns
        -------
        response_no : int
            0: ebmedding exsists with same faceid
            1: ebmedding saved
        """
        # Detect 1 face in frame
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            keep_all=False, device=self.device
            )
        # Crop 1 face
        img_cropped = mtcnn(frame)
        # Get embedding (unsqueeze to add batch dimension)
        embedding = self.resnetv1(img_cropped.unsqueeze(0).to(self.device))
        np_embedding = embedding.squeeze().to('cpu').detach().numpy().copy()
        # Remove mtcnn to reduce GPU memory usage
        del mtcnn
        
        # Check latest embedding file date
        faceid_path = os.path.join(EMBEDDINGS, faceid)
        file_num, file_date = fdu.get_newest_npy_file_date(faceid_path)

        response_no = 0
        # To avoid duplicate registration, set interval. 
        now = datetime.now()
        if now - file_date > timedelta(seconds = 5):
            # Save embedding
            self.save_embedding(faceid, np_embedding)
            response_no = 1
        
        return response_no

    def reset_faiss(self):     
        # Reset Faiss
        self.index.reset()
        logger.info("Faiss index reset.")
        # Init Faiss
        self.index, self.index_ids = self.init_faiss()

    def delete_faceid(self, faceid):
        """
        Parameters
        ----------
        faceid : str
        """
        faceid_path = os.path.join(EMBEDDINGS, faceid)
        # Delete faceid and all embedding files
        result = fdu.delete_faceid_files(faceid_path)
        
        # Reset Faiss index
        if result == 1:
            logger.info("Directory deleted: %s" % faceid_path)
            # Reset Faiss
            self.faiss_event_manager("reset")

        return result
    
    # Faiss event manager(Text file version).
    def faiss_event_manager_text(self):
        """
        Command Option
        ADD /data/embedding_root/faceid/XXX.npy
        RESET
        """
        # set reader index
        read_idx = 0
        
        while True:
            with self.faiss_queue.open(mode='r') as q:
                faiss_queue = q.readlines()
            # if queue is bigger than read index, proceed process
            lines = len(faiss_queue)
            if  lines > read_idx:
                for line in faiss_queue[read_idx:lines]:
                    # strip text
                    try:
                        cmd = line.split()[0]
                    except IndexError:
                        pass
                    
                    # Process command
                    if cmd == "ADD":
                        npy_file = line.split()[1]
                        if os.path.exists(npy_file):
                            self.add_embedding(npy_file)                            
                    elif cmd == "RESET":
                        self.reset_faiss()
                # Renew index position
                read_idx = lines
            # sleep thread a while
            time.sleep(1)

    # Faiss event manager(Redis version).
    def faiss_event_manager_redis(self):
        """
        List Name: Value, ...
        ADD: /data/embedding_root/faceid/XXX.npy, ...
        RESET: RESET, RESET, ....
        """
        # set reader index
        add_idx = 0
        reset_idx = 0
        reset_wait = 0
        # reddis db connection
        r = fdu.connect_redis()
        while True:
            # Get 'ADD' list on redis
            lines = r.lrange('ADD', add_idx, -1)
            if lines:
                for line in lines:
                    npy_file = line.decode()
                    if os.path.exists(npy_file):
                        self.add_embedding(npy_file)
                # Renew index position
                add_idx += len(lines)
            # Get 'RESET' list on redis    
            lines = r.lrange('RESET', reset_idx, -1)
            if lines:
                # Even if there're multiple reset request, process 1 at a time.
                if reset_wait == 0:
                     self.reset_faiss()
                     # Renew index position
                     reset_idx += len(lines)
                     # set reset interval
                     reset_wait = 10
                     
            # sleep thread a while
            time.sleep(1)
            # count down reset interval
            if reset_wait > 0:
                reset_wait -= 1

    def faiss_event_manager(self, cmd, **kwargs):
        mode = FAISS_EVENT_MANAGER
        npy_name = kwargs.get("npy_name")
        
        if mode == 'redis':
            if cmd == 'init':
                logger.info("Faiss event manager started: Redis")
                # Initialize db
                self.faiss_db = fdu.connect_redis()
                self.faiss_db.flushdb()
                # start event manager
                threading.Thread(target=self.faiss_event_manager_redis, daemon=True).start()
            elif cmd =='add':
                self.faiss_db.rpush('ADD', npy_name)
            elif cmd == 'reset':
                self.faiss_db.rpush('RESET', 'RESET')
            
        elif mode == 'text':
            if cmd == 'init':
                logger.info("Faiss event manager started: Text")
                # Initialize queue file
                self.faiss_queue = Path(FAISS_QUEUE)
                if not self.faiss_queue.exists():
                    os.makedirs(os.path.dirname(self.faiss_queue), exist_ok=True)
                    self.faiss_queue.touch()
                elif len(self.faiss_queue.open(mode='r').readlines()) > 0 :
                    self.faiss_queue.unlink(missing_ok=True)
                    self.faiss_queue.touch()
                # start event manager
                threading.Thread(target=self.faiss_event_manager_text, daemon=True).start()
            elif cmd =='add':
                with self.faiss_queue.open(mode='a') as q:
                    q.write('ADD ' + npy_name + '\n')
            elif cmd == 'reset':
                with self.faiss_queue.open(mode='a') as q:
                    q.write('RESET\n')
            
        else:
            if cmd == 'init':
                logger.warn("Faiss event manager is not enabled.")
            elif cmd =='add':
                self.add_embedding(npy_name)
            elif cmd == 'reset':
                self.reset_faiss

    """
    For API
    """
    def get_faces(self):
        faces = []
        count_dict = collections.Counter(self.index_ids)
        for k, v in count_dict.items():
            face = far.Face()
            face.faceid = k
            face.numfiles = v
            try: 
                face.groupid = k.split('-')[0]
                face.userid = k.split('-')[1]
            except IndexError:
                face.userid = face.faceid       
            faces.append(face)

        return faces
    
    def get_face(self, faceid):
        face = far.Face()
        count_dict = collections.Counter(self.index_ids)
        v = count_dict[faceid]
        if v:
            face.faceid = faceid
            face.numfiles = v
            try: 
                face.groupid = faceid.split('-')[0]
                face.userid = faceid.split('-')[1]
            except IndexError:
                face.userid = face.faceid
    
        return face

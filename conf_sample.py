#################
# API Configuration
#################
# For uvicorn
LOG_OUTPUT = 'uvicorn'
# For gunicorn
#LOG_OUTPUT = 'gunicorn.error'
# Log level
LOG_LEVEL = 'DEBUG'

# Fast Api
API_TITLE = 'Face Recognition API'
API_VERSION = '1.0.0'

##################
# Face Recognition
##################
# Enable GPU
ENABLE_GPU = True

# Capture calculation speed (Face detection, Embedding, Index Search)
SPEED_TEST = False

# Embedding Files
EMBEDDINGS = '/var/lib/aoiface/embeddings'

# Pretrained model cache directory
CACHE_DIR = '/var/lib/aoiface/torch'

# When embedding distance below this, show comapny id. (Recommend:0.1 ~ 0.6)
#  (0.0 = identical, 1.0 < different)
IDENTICAL_DISTANCE = 0.49

# When embedding distance below this, save embedding (Recomend: 0.1 ~ 0.3 )
SAVE_DISTANCE = 0.3

# Embedding save days interval (Recomend: 7 days)
SAVE_INTERVAL = 2

# Minimum capture embedding files (Recomend: 2-3)
# Below this, save embedding regardless SAVE_INTERVAL
SAVE_MINIMUM = 2

##################
# Config FAISS
##################
# Type of index serch (Recomend: False)
# If person > 1,000 and image >10,000, set True for IndexIVFFlat
IVFFLAT = False

# Faiss event manager
# When gunicorn worker is more than 2, use text or redis for thread safe.
#FAISS_EVENT_MANAGER = ''
#FAISS_EVENT_MANAGER = 'text'
FAISS_EVENT_MANAGER = 'redis'

# When use 'text'
#FAISS_QUEUE = '/var/lib/aoiface/faiss_queue.txt'
FAISS_QUEUE = ''

# Redis index
FAISS_IDX = '1'

##################
# Config for writing results to redis
##################
# If True, write detection results on redis
ENABLE_DETECT_DB = False

# Redis index
DETECT_IDX = '2'

# Detection result expire on redis (Recomend: 1 ~ 3 sec)
DETECT_EXPIRE = 3

##################
# Redis
##################
REDIS_HOST = 'localhost'
REDIS_PORT =  '6379'
REDIS_PASS = 'password'

##################
# Database
##################
DBUSER = "dbuser"
DBPASS = "dbpass"
DBHOST = "127.0.0.1"
DBPORT = 3306
DBNAME = "face_db"
ENABLE_PROCEDURE = True

##################
# Auth Organizer
##################
# Continuous same combination of face id and device id
# are authorized only 1 time in auth period(sec).
AUTH_PERIOD = 60

# 1 nearest distance threshold for authorize
AUTH_1 = 0.2

# Average of 3 nearest distance threshold for authorize
AUTH_2 = 0.35

# Logging
ORGANIZER_LOG_LEVEL = 'DEBUG'
ORGANIZER_LOGFILE = '/var/log/aoiface/auth_organaizer_log.txt'

##################
# Batch
##################
# Original image file directory
PICTURES = '/var/lib/aoiface/original'

# Embedding file output directory
EMBEDDINGS_BATCH_OUT = '/var/lib/aoiface/embeddings'

# Batch mode or non-batch mode
BATCH_MODE = True

# Image size
PIC_WIDTH = 472
PIC_HIGHT = 579

# Number of Worker
WORKERS = 2

# Logging
BATCH_LOG_LEVEL = 'DEBUG'
BATCH_LOGFILE = '/var/log/aoiface/batch_log.txt'
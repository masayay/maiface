#
# Gunicorn config file
#
wsgi_app = 'face_api:app'

# Server Mechanics
#========================================
chdir = '/var/www/maiface'
user = 'gunicorn'
group = 'gunicorn'
timeout = '300'

# Server Socket
#========================================
# When using unix socket, access log cannot get 'access from ip' 
#bind = 'unix:gunicorn.sock'
bind = '127.0.0.1:8000'

# Worker Processes
#========================================
workers = 2
worker_class = 'uvicorn.workers.UvicornWorker'

#  Logging
#========================================
# access log
#access_log_format = '%({x-forwarded-for}i)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
accesslog = '/var/log/gunicorn/maiface_access.log'
# gunicorn log
errorlog = '/var/log/gunicorn/maiface_error.log'
loglevel = 'info'
capture_output = True

# Face Recognition Web Server
Developed with Pytorch, Faiss and Fastapi.

## Demo
After register your face, return group name and user id.  
![demo2a](https://user-images.githubusercontent.com/92005636/162384252-1dfacef8-1c6c-4a01-bc38-6fd0bf905248.jpg)  

## Install
1. Instal requirements  
~~~
apt install python3-pip
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install faiss-gpu
pip3 install facenet-pytorch
pip3 install aiofiles Jinja2
pip3 install pillow
pip3 install fastapi python-multipart
pip3 install uvicorn[standard] Gunicorn
~~~
2. Configure Application  
~~~
mkdir /var/www/
cd /var/www
git clone https://github.com/masayay/maiface.git
mv maiface/conf_linux_sample.py maiface/conf.py
~~~
3. Configure Gunicorn
~~~
mkdir /etc/gunicorn
mv maiface/maiface_config.py /etc/gunicorn/maiface_config.py

mkdir /var/log/gunicorn
mkdir /var/lib/maiface
mkdir /var/lib/maiface/cache
mkdir /var/lib/maiface/embeddings

useradd -U -m -s /usr/sbin/nologin gunicorn
chown gunicorn:gunicorn /var/log/gunicorn
chown -R gunicorn:gunicorn /var/www/maiface
chown -R gunicorn:gunicorn /var/lib/maiface
chown -R gunicorn:gunicorn /etc/gunicorn
~~~
4. Start Application
~~~
mv maiface/systemd_sample.txt /etc/systemd/system/maiface.service
systemctl daemon-reload
systemctl start maiface
~~~
5. Start nginx
~~~
apt install nginx
cp maiface/nginx_sample.txt /etc/nginx/sites-available/maiface
rm -f /etc/nginx/sites-enabled/default
ln -s /etc/nginx/sites-available/maiface /etc/nginx/sites-enabled/maiface
systemctl start nginx
~~~

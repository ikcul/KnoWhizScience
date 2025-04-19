# Pipeline Service

## Installation

```bash
conda create --name knowhiz python=3.11
conda activate knowhiz
# pip install langchain openai unstructured pdf2image pdfminer pdfminer.six "langchain[docarray]" tiktoken scipy faiss-cpu pandas pymupdf langchain_openai langchain_community langchain-anthropic scikit-learn
# pip install quart quart-cors celery "celery[redis]" gevent eventlet pymongo azure-core azure-storage-blob
# pip install discord.py
# pip install moviepy pydub wikipedia wikipedia-api youtube-transcript-api
pip install -r requirements.txt
```

install MacTex or TeX Live

```bash
# e.g. on macOS or Linux
brew install --cask mactex
```

install ffmpeg

```bash
# e.g. on macOS or Linux
brew install ffmpeg
```

Once installed, you can set the IMAGEIO_FFMPEG_EXE environment variable as indicated in your script. This variable points to the FFmpeg executable, which is typically located in /usr/local/bin/ffmpeg on macOS, but the provided script suggests a Homebrew-specific path under /opt/homebrew/bin/ffmpeg. Verify the correct path using:

```bash
which ffmpeg
```

Then update the environment variable accordingly in your Python script or set it in your shell profile:

```bash
export IMAGEIO_FFMPEG_EXE=$(which ffmpeg)
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
```

## Set OPENAI_API_KEY

```bash
cd knowhizService
# Should replace sk-xxx to a real openai api key
echo "OPENAI_API_KEY=sk-xxx" > .env
```

## Run Native

```bash
# Copy the pdf file to knowhizService/pipeline/test_inputs/ folder
conda activate knowhiz
cd knowhizService
python local_test.py <filename>
```

## Run Service

```bash
# Run this command if redis server not running
SHELL1: $ redis-server &

# Start celery task
SHELL1: $ conda activate knowhiz
SHELL1: $ cd knowhizService
SHELL1: $ celery -A pipeline.tasks worker --loglevel=info --pool=gevent --concurrency=100 --logfile log.log

# Start RESTful service
SHELL2: $ conda activate knowhiz
SHELL2: $ cd knowhizService
SHELL2: $ python -m pipeline.app
```

## Deploy in AWS

### Prerequsites

Install redis-server

```bash
# Ref: https://feliperohdee.medium.com/installing-redis-to-an-aws-ec2-machine-2e2c4c443b68

# Update yum packages and install
sudo yum -y update
sudo yum -y install gcc gcc-c++ make wget git curl tmux tree

# Download and install Redis, change to the latest version
cd ~/Downloads/
wget http://download.redis.io/releases/redis-4.0.9.tar.gz
tar xzf redis-4.0.9.tar.gz
rm redis-4.0.9.tar.gz

# Recompile Redis
cd redis-4.0.9
make distclean
make

# Install TCL and test Redis
sudo yum install -y tcl
make test

# Make base directories and copy config files to them
sudo mkdir /etc/redis
sudo chown ec2-user:ec2-user /etc/redis
sudo cp src/redis-server src/redis-cli /usr/local/bin
sudo cp redis.conf /etc/redis/redis.conf

# sudo nano /etc/redis/redis.conf
dir ./ -> dir /etc/redis
daemonize no -> daemonize yes
pidfile /var/run/redis.pid -> pidfile /etc/redis/redis.pid
logfile '' -> logfile /etc/redis/redis_log

# Download a boot script and add it to "init.d" folder, turn it executable, and allow system to auto start it
cd ~/Downloads
sudo wget https://gist.githubusercontent.com/feliperohdee/d04126b0b727e2a0ef5eee04542794df/raw/4531d1809639fe00bef81985fe076f6a004471be/redis-server
sudo mv redis-server /etc/init.d
sudo chmod 755 /etc/init.d/redis-server
sudo chkconfig --add redis-server
sudo chkconfig --level 345 redis-server on

# ...

# Append this file to fix a low-memory issue on backups, and disable memory swap (which could cause Redis’ process to be blocked by the I/O operation of the disk)
sudo nano /etc/sysctl.conf
# Add
vm.overcommit_memory = 1
vm.swappiness = 0

# Finally, start the service, and test it
sudo service redis-server start
redis-cli ping
#PONG
```

### Setup environment

```bash
# Install anaconda
sudo yum -y install libXcomposite libXcursor libXi libXtst libXrandr alsa-lib mesa-libEGL libXdamage mesa-libGL libXScrnSaver
cd ~/Downloads
curl -O https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash Anaconda3-2023.09-0-Linux-x86_64.sh

# Add below in ~/.bash_profile to enable conda
eval "$(/home/ec2-user/anaconda3/bin/conda shell.bash hook)"

# Run below command once. Disable conda auto activate base env
conda config --set auto_activate_base false

# Setup conda environment
conda create --name knowhiz python=3.11
# Activate anaconda
conda activate knowhiz

# Download pipeline repo
cd ~/
git clone https://github.com/CuraStone/KnoWhizPipeline.git

# Install required python packages
cd KnoWhizPipeline
pip install -r knowhizService/requirements.txt

# Set OPENAI_API_KEY
# Should replace sk-xxx to a real openai api key
echo "OPENAI_API_KEY=sk-xxx" > knowhizService/.env
```

### Start pipeline service

With Tmux

```bash
# Start celery task
SHELL1: $ conda activate knowhiz
SHELL1: $ export OPENAI_API_KEY=sk-xxx
SHELL1: $ cd ~/KnoWhizPipeline/knowhizService
SHELL1: $ celery -A pipeline.tasks worker --loglevel=info --pool=gevent --concurrency=100 --logfile log.log

# Start RESTful service
SHELL2: $ conda activate knowhiz
SHELL2: $ export OPENAI_API_KEY=sk-xxx
SHELL2: $ cd ~/KnoWhizPipeline/knowhizService
SHELL2: $ python -m pipeline.app
```

### Set AWS Route53

Set A record "pipeline.knowhiz.us" to this EC2 IP.

### Set Caddyfile

```bash
# Modify ~/Caddyfile by adding:
pipeline.knowhiz.us {
  reverse_proxy localhost:8081
}

# Restart caddy
cd ~/
sudo caddy stop
sudo caddy start
```

### Issue fix

#### Unsupported compiler -- at least C++11 support is needed

```bash
sudo yum -y install gcc-c++
```

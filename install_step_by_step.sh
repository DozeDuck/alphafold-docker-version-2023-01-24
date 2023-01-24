conda create -n af2 python=3.9 -y
conda activate af2

pip install -r requirements.txt

pip install --upgrade jax
# pip install --upgrade jax==0.2.14 jaxlib==0.1.69

conda install -c conda-forge openmm=7.5.1
# 验证openmm：python -m simtk.testInstallation
"""
直接执行python -m openmm.testInstallation 会报错找不到openmm

输出以下内容就说明安装好了。

OpenMM Version: 7.5.1 Git Revision:

There are 4 Platforms available:

1 Reference - Successfully computed forces 2 CPU - Successfully computed forces 3 CUDA - Successfully computed forces 4 OpenCL - Successfully computed forces

Median difference in forces between platforms:

Reference vs. CPU:  Reference vs. CUDA: CPU vs. CUDA: Reference vs. OpenCL:  CPU vs. OpenCL: CUDA vs. OpenCL:

All differences are within tolerance.

如果输出内容里出现：CUDA - Error computing forces with CUDA platform，原因在于cudatookit不对。首先使用 nvidia-smi 查看CUDA Version，然后使用 conda install -c conda-forge cudatoolkit= CUDA Version(对应的版本号)，就可以解决问题。当然没有cuda也能跑起来，只是费时间。
"""

conda install -y -c conda-forge pdbfixer

# 验证Alphafold：
python run_alphafold_test.py
"""
出现以下内容，就说明安装好了。

[ RUN  ] RunAlphafoldTest.test_end_to_end_no_relax I0814 21:46:41.874690 140372256589632 run_alphafold.py:161] Predicting test I0814 21:46:41.875205 140372256589632 run_alphafold.py:190] Running model model1 on test I0814 21:46:41.875339 140372256589632 run_alphafold.py:202] Total JAX model model1 on test predict time (includes compilation time, see --benchmark): 0.0s I0814 21:46:41.878139 140372256589632 run_alphafold.py:271] Final timings for test: {'features':
3.409385681152344e-05, 'process_features_model1': 3.838539123535156e-05, 'predict_and_compile_model1': 2.0742416381835938e-05} [ OK  ] RunAlphafoldTest.test_end_to_end_no_relax [ RUN  ] RunAlphafoldTest.test_end_to_end_relax I0814 21:46:41.880331 140372256589632 run_alphafold.py:161] Predicting test I0814 21:46:41.880626 140372256589632 run_alphafold.py:190] Running model model1 on test I0814 21:46:41.880749 140372256589632 run_alphafold.py:202] Total JAX
model model1 on test predict time (includes compilation time, see --benchmark): 0.0s I0814 21:46:41.883405 140372256589632 run_alphafold.py:271] Final timings for test: {'features': 3.0994415283203125e-05, 'process_features_model1': 3.409385681152344e-05, 'predict_and_compile_model1': 1.6450881958007812e-05, 'relax_model1': 2.9087066650390625e-05}

[ OK  ] RunAlphafoldTest.test_end_to_end_relax

Ran 2 tests in 0.011s

OK
"""

conda install -c bioconda aria2

# install Docker & Nvidia container toolkit
# Uninstall old versions
# Older versions of Docker went by the names of docker, docker.io, or docker-engine. Uninstall any such older versions before attempting to install a new version:
sudo apt-get remove docker docker-engine docker.io containerd runc

# Set up the repository
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg lsb-release

# Add Docker#s official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Use the following command to set up the repository:
echo  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
# Receiving a GPG error when running apt-get update?
# Your default umask may be incorrectly configured, preventing detection of the repository public key file. Try granting read permission for the Docker public key file before updating the package index:
# sudo chmod a+r /etc/apt/keyrings/docker.gpg
# sudo apt-get update

# Install the latest version
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Activate the docker engine
sudo service docker status
sudo service docker start
sudo docker run hello-world

# Non-root user using docker
sudo groupadd docker
sudo usermod -aG docker dozeduck
sudo service docker restart
# newgrp docker
# restart the terminal then try
docker run hello-world

# Install Nvidia Container Toolkit
nvidia-smi
docker -v
more install_commands.sh | grep docker
curl https://get.docker.com | sh   && sudo systemctl --now enable docker
docker run hello-world
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)  && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg  && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list |             sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |             sudo tee
/etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y  nvidia-docker2
sudo systemctl restart docker
docker run hello-world
tail install_commands.sh
sudo service docker restart
docker run hello-world
tail -20 install_commands.sh
sudo service docker start
docker run hello-world
docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
sudo docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi

# using docker build AlphaFold image
docker build -f docker/Dockerfile -t alphafold .
# Just in case you want to remove this image:
docker images -a
docker stop $(docker ps -a -q)
docker rmi <your-image-id>

# may need a conda enviroment, but actually if you already done the first part of this introduction,
# I mean if you have created "af2" then you don't need this new env.
# conda create -n af2docker python=3.9 -y
pip3 install -r docker/requirements.txt

# Download database
cd ..
mkdir database_alphafold
sudo mount -t drvfs G: /home/dozeduck/workspace/database_alphafold
sudo umount /home/dozeduck/workspace/database_alphafold
cd /home/dozeduck/workspace/database_alphafold
bash ../alphafold/script/download_all_data.sh /home/dozeduck/workspace/database_alphafold
# 若报错缺少libssl.so.1.0.0则运行以下命令
sudo apt-get update
sudo apt-get install libssl1.0.0 libssl-dev
bash ../alphafold/script/download_all_data.sh /home/dozeduck/workspace/database_alphafold
# 若报错tar: params_model_1.npz: Cannot utime: Operation not permitted
sudo bash ../alphafold/script/download_all_data.sh /home/dozeduck/workspace/database_alphafold
# sudo chmod 755 --recursive "$DOWNLOAD_DIR"

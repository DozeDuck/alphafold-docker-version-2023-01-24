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

################################################################################################################################
预测wip1模型
nvidia-smi
(af2) dozeduck@DozeDuck-R12:~/workspace/database_alphafold/9.AF_test$ nvidia-smi
Sat Jan 28 00:18:27 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.65.01    Driver Version: 516.94       CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:02:00.0  On |                  N/A |
|  0%   36C    P8    22W / 350W |   1351MiB / 24576MiB |     28%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A        73      G   /Xorg                           N/A      |
|    0   N/A  N/A        94      G   /xfce4-session                  N/A      |
|    0   N/A  N/A       155      G   /xfwm4                          N/A      |
|    0   N/A  N/A       415      G   /Xwayland                       N/A      |
+-----------------------------------------------------------------------------+

conda activate af2
sudo service docker start
python ../docker/run_docker.py --fasta_paths=wip1.fasta --max_template_date=2022-01-01 --data_dir=$AFDB --output_dir=$PWD

##################################################################################################################################

例子

下面举例说明如何在不同场景下使用 AlphaFold。



折叠单体

假设我们有一个带有序列的单体<SEQUENCE>。输入 fasta 应该是：



>sequence_name

<SEQUENCE>

然后运行以下命令：



python3 docker/run_docker.py \

  --fasta_paths=monomer.fasta \

  --max_template_date=2021-11-01 \

  --model_preset=monomer \

  --data_dir=$DOWNLOAD_DIR \

  --output_dir=/home/user/absolute_path_to_the_output_dir

折叠均聚体

假设我们有一个具有相同序列的 3 个副本的同聚体<SEQUENCE>。输入 fasta 应该是：



>sequence_1

<SEQUENCE>

>sequence_2

<SEQUENCE>

>sequence_3

<SEQUENCE>

然后运行以下命令：



python3 docker/run_docker.py \

  --fasta_paths=homomer.fasta \

  --max_template_date=2021-11-01 \

  --model_preset=multimer \

  --data_dir=$DOWNLOAD_DIR \

  --output_dir=/home/user/absolute_path_to_the_output_dir

折叠异聚体

假设我们有一个 A2B3 异聚体，即有 2 个拷贝<SEQUENCE A>和 3 个拷贝<SEQUENCE B>。输入 fasta 应该是：



>sequence_1

<SEQUENCE A>

>sequence_2

<SEQUENCE A>

>sequence_3

<SEQUENCE B>

>sequence_4

<SEQUENCE B>

>sequence_5

<SEQUENCE B>

然后运行以下命令：



python3 docker/run_docker.py \

  --fasta_paths=heteromer.fasta \

  --max_template_date=2021-11-01 \

  --model_preset=multimer \

  --data_dir=$DOWNLOAD_DIR \

  --output_dir=/home/user/absolute_path_to_the_output_dir

一个接一个折叠多个单体

假设我们有两个单体，monomer1.fasta和monomer2.fasta。



我们可以使用以下命令依次折叠两者：



python3 docker/run_docker.py \

  --fasta_paths=monomer1.fasta,monomer2.fasta \

  --max_template_date=2021-11-01 \

  --model_preset=monomer \

  --data_dir=$DOWNLOAD_DIR \

  --output_dir=/home/user/absolute_path_to_the_output_dir

一个接一个地折叠多个多聚体

假设我们有两个多聚体，multimer1.fasta并且multimer2.fasta。



我们可以使用以下命令依次折叠两者：



python3 docker/run_docker.py \

  --fasta_paths=multimer1.fasta,multimer2.fasta \

  --max_template_date=2021-11-01 \

  --model_preset=multimer \

  --data_dir=$DOWNLOAD_DIR \

  --output_dir=/home/user/absolute_path_to_the_output_dir

AlphaFold 输出

输出将保存在通过 （默认为）--output_dir标志提供的目录的子目录中。输出包括计算的 MSA、未松弛结构、松弛结构、排名结构、原始模型输出、预测元数据和节时序。该目录将具有以下结构：run_docker.py/tmp/alphafold/--output_dir



<target_name>/

    features.pkl

    ranked_{0,1,2,3,4}.pdb

    ranking_debug.json

    relax_metrics.json

    relaxed_model_{1,2,3,4,5}.pdb

    result_model_{1,2,3,4,5}.pkl

    timings.json

    unrelaxed_model_{1,2,3,4,5}.pdb

    msas/

        bfd_uniref_hits.a3m

        mgnify_hits.sto

        uniref90_hits.sto

每个输出文件的内容如下：



features.pkl–pickle包含模型用于生成结构的输入特征 NumPy 数组的文件。



unrelaxed_model_*.pdb– 包含预测结构的 PDB 格式文本文件，与模型输出的完全相同。



relaxed_model_*.pdb– 包含预测结构的 PDB 格式文本文件，在对未松弛结构预测执行 Amber 松弛程序后（有关详细信息，请参见 Jumper 等人 2021 年，补充方法 1.8.6）。



ranked_*.pdb– 在按模型置信度重新排序后，包含松弛预测结构的 PDB 格式文本文件。这里ranked_0.pdb应该包含置信度最高ranked_4.pdb的预测和置信度最低的预测。为了对模型置信度进行排名，我们使用预测的 LDDT (pLDDT) 分数（有关详细信息，请参阅 Jumper 等人 2021 年增刊方法 1.9.6）。



ranking_debug.json– 一个 JSON 格式的文本文件，其中包含用于执行模型排名的 pLDDT 值，以及返回原始模型名称的映射。



relax_metrics.json– 包含放宽指标的 JSON 格式文本文件，例如剩余违规。



timings.json– 一个 JSON 格式的文本文件，其中包含运行 AlphaFold 管道的每个部分所花费的时间。



msas/- 包含描述用于构建输入 MSA 的各种遗传工具命中的文件的目录。



result_model_*.pkl–pickle包含模型直接生成的各种 NumPy 数组的嵌套字典的文件。除了结构模块的输出外，这还包括辅助输出，例如：



Distograms（distogram/logits包含形状为 [N_res, N_res, N_bins] 的 NumPy 数组并distogram/bin_edges包含 bin 的定义）。

每个残基 pLDDT 分数（plddt包含一个形状为 [N_res] 的 NumPy 数组，可能值的范围从0到100，其中100 表示最有信心）。这可以用于识别以高置信度预测的序列区域，或作为每个目标的总体置信度得分，当对残基进行平均时。

仅在使用 pTM 模型时存在：预测的 TM 分数（ptm字段包含标量）。作为全局叠加指标的预测指标，该分数还旨在评估模型是否对整体域打包有信心。

仅在使用 pTM 模型时存在：预测的成对对齐错误（predicted_aligned_error包含形状为 [N_res, N_res] 的 NumPy 数组，可能值的范围从0到 max_predicted_aligned_error，其中0表示最有信心）。这可以用于结构内域打包置信度的可视化。

pLDDT 置信度存储在输出 PDB 文件的 B 因子字段中（尽管与 B 因子不同，pLDDT 越高越好，因此在用于分子替换等任务时必须小心）。



此代码已经过测试，可以匹配 CASP14 测试集上的平均前 1 准确度，其中 pLDDT 对 5 个模型预测进行排名（一些 CASP 目标使用早期版本的 AlphaFold 运行，一些有手动干预；有关详细信息，请参阅我们即将出版的出版物）。某些目标（例如 T1064）也可能与随机种子相比具有较高的个体运行方差。

ARG BASE_IMAGE=nvidia/cuda:12.1.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libopencv* \
    libvulkan* \
    git \
    python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip "setuptools<82" wheel && \
    pip install numpy==1.24.4


RUN git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules
WORKDIR /SimplerEnv/ManiSkill2_real2sim
RUN pip install -r requirements.txt  && \
    pip install -e .

WORKDIR /SimplerEnv 
RUN pip install -r requirements_full_install.txt  && \
    pip install -e .
RUN apt update && apt install -y libvulkan1 vulkan-tools ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip install tensorflow==2.15.0 && \
    pip install tensorflow[and-cuda]==2.15.1 && \
    pip install git+https://github.com/nathanrooy/simulated-annealing

# install gsutil
RUN apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg curl && \
    rm -rf /var/lib/apt/lists/*
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y && rm -rf /var/lib/apt/lists/*

# install headless util unzip 
RUN apt-get update && apt-get install -y xvfb unzip && \
    rm -rf /var/lib/apt/lists/*


# download weights
RUN gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_tf_trained_for_002272480_step.zip . && \
    unzip rt_1_x_tf_trained_for_002272480_step.zip && \
    mv rt_1_x_tf_trained_for_002272480_step rt_1_checkpoints && \
    rm rt_1_x_tf_trained_for_002272480_step.zip
# RUN gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_tf_trained_for_000400120 rt_1_400k_checkpoints
# RUN gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_tf_trained_for_000058240 rt_1_58k_checkpoints    
# RUN gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_tf_trained_for_000001120 rt_1_1k_checkpoints    

# install octo
RUN pip install --upgrade "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 
RUN bash -c "git clone https://github.com/octo-models/octo/ ; cd octo; git checkout 653c54acde686fde619855f2eac0dd6edad7116b  ;pip install -e ."
RUN pip install transformers==4.34.1

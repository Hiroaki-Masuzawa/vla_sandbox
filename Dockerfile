FROM nvidia/cuda:12.1.1-devel-ubuntu22.04


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

RUN pip install tensorflow==2.15.0
# RUN pip install -r requirements_full_install.txt
RUN pip install tensorflow[and-cuda]==2.15.1
RUN pip install git+https://github.com/nathanrooy/simulated-annealing

RUN apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg curl
# RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
# RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
# RUN apt-get update && sudo apt-get install google-cloud-cli
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y

RUN apt-get update && apt-get install -y unzip 

RUN gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_tf_trained_for_002272480_step.zip . && \
    unzip rt_1_x_tf_trained_for_002272480_step.zip && \
    mv rt_1_x_tf_trained_for_002272480_step checkpoints && \
    rm rt_1_x_tf_trained_for_002272480_step.zip

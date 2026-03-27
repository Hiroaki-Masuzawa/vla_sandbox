ARG BASE_IMAGE=nvidia/cuda:12.1.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    LANG=en_US.UTF-8

# ---- Base setup ----
RUN apt-get update && apt-get install -y \
    locales \
    software-properties-common \
    curl \
    git \
    python3-pip \
    ffmpeg \
    mesa-utils \
    && locale-gen en_US en_US.UTF-8 \
    && add-apt-repository universe \
    && rm -rf /var/lib/apt/lists/*

# ---- ROS2 setup ----
RUN ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
    | grep -F "tag_name" | awk -F'"' '{print $4}') && \
    curl -L -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb" && \
    dpkg -i /tmp/ros2-apt-source.deb

RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-joint-state-publisher-gui \
    ros-humble-controller-manager \
    ros-humble-ros2-controllers \
    ros-humble-joint-state-broadcaster \
    ros-humble-cv-bridge \
    ros-humble-rqt-image-view \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
RUN pip install --no-cache-dir \
    "mujoco>=3.3.7,<4" \
    "lerobot[all]"

# ---- Workspace ----
WORKDIR /workspace
RUN git clone https://github.com/FumitakaIwaki/ros2-so101-mujoco.git

# ---- Patch ----
WORKDIR /workspace/ros2-so101-mujoco
RUN <<EOF
cat << 'PATCH' | patch -p1
diff --git a/ros2_ws/src/lerobot_vla/lerobot_vla/smolvla_node.py b/ros2_ws/src/lerobot_vla/lerobot_vla/smolvla_node.py
index db4b303..fbf1427 100644
--- a/ros2_ws/src/lerobot_vla/lerobot_vla/smolvla_node.py
+++ b/ros2_ws/src/lerobot_vla/lerobot_vla/smolvla_node.py
@@ -34,7 +34,7 @@ class SmolVLANode(Node):
         self.action_rate = self.get_parameter('action_rate').value

         # VLA設定
-        self.device = torch.device("mps")
+        self.device = torch.device("cuda")
         self.model_id = MODEL_ID
         self.model = SmolVLAPolicy.from_pretrained(self.model_id)
         self.preprocess, self.postprocess = make_pre_post_processors(
PATCH
EOF

# ---- Build ----
WORKDIR /workspace/ros2-so101-mujoco/ros2_ws
RUN bash -c "source /opt/ros/humble/setup.bash && colcon build"
xhost +local:docker

# コンテナ起動
docker run --rm -it \
  --shm-size=16gb \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $XDG_RUNTIME_DIR:$XDG_RUNTIME_DIR \
  -v /usr/share/vulkan:/usr/share/vulkan:ro \
  -v /etc/vulkan:/etc/vulkan:ro \
  -v $(pwd):/userdir \
  -w /userdir \
    vla-ros2  bash



  # --gpus '"device=GPU-fc0fabc3-1085-fc36-1739-60385c5d33cc"' \
#   -u `id -u`:`id -g` \
#   -v /etc/passwd:/etc/passwd:ro \
#   -v /etc/group:/etc/group:ro \
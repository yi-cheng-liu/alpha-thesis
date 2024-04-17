# Get the user account information to an intermediate file
getent passwd $(id --user) > $HOME/.$(id --user).passwd
getent group $(id --groups) > $HOME/.$(id --user).group

docker build -t wanhanxi/cca:tf .

# Mount user information and project folder for running
docker run -itd                                         \
     --user $(id --user):$(id --group)                  \
     $(id --groups | sed 's/\(\b\w\)/--group-add \1/g') \
     -v $HOME/.$(id --user).passwd:/etc/passwd:ro       \
     -v $HOME/.$(id --user).group:/etc/group:ro         \
     -e DISPLAY                                         \
     --gpus all                                         \
     -v /tmp/.X11-unix/:/tmp/.X11-unix/                 \
     -v /usr/bin/prime-run:/usr/bin/prime-run           \
     -v ~/.Xauthority:/root/.Xauthority:rw              \
     -v .:/workspace/cca                             \
     -v ~/temp:/home/$(whoami)                          \
     --privileged                                       \
     --network host                                     \
     --name whxccatf wanhanxi/cca:tf

# If fails, use the following command to run the container
# docker run -itd \
#          -e DISPLAY \
#          --gpus all \
#          -v /tmp/.X11-unix/:/tmp/.X11-unix/  \
#          -v /usr/bin/prime-run:/usr/bin/prime-run \
#          -v ~/.Xauthority:/root/.Xauthority \
#          -v .:/workspace/pytorch_td_rex \
#          --privileged \
#          --network host \
#          --name tdrex wanhanxi/tdrex
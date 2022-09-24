FROM gitpod/workspace-full

RUN sudo apt-get update \
    && sudo apt-get install -y \
    tool \
    && libxcb-image -y \
    &&  libxcb-icccm4 -y \
    && libxcb-image0 -y \
    && libxcb-keysyms1 -y \
    &&   libxcb-randr0 -y \
    && libxcb-render-util0 -y \
    &&  libxcb-shape0 -y \
    &&  libxcb-xfixes0 -y \
    && libxcb-xinerama0 -y \
    &&  libxcb-xkb1 -y \
    && libxkbcommon-x11-0 \
    && libxcb-render-util -y \
    python3 smb_ai.py --load-file "Example world1-1"  --load-inds 1213 \

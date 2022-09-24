FROM gitpod/workspace-full

RUN sudo apt-get update \
 && sudo apt-get install -y \
    tool \
 && sudo rm -rf /var/lib/apt/lists/*
   
    sudo apt-get install libxcb-image -y \

    sudo apt-get install libxcb-icccm4 -y \

    sudo apt-get install libxcb-image0 -y \

    sudo apt-get install libxcb-keysyms1 -y \

    sudo apt-get install libxcb-randr0 -y \

    sudo apt-get install libxcb-render-util0 -y \

    sudo apt-get install libxcb-shape0 -y \

    sudo apt-get install libxcb-xfixes0 -y \

    sudo apt-get install libxcb-xinerama0 -y \

    sudo apt-get install libxcb-xkb1 -y \

    sudo apt-get install -y libxkbcommon-x11-0 \

    sudo apt-get libxcb-render-util -y \

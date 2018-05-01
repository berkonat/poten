#!/bin/bash
#TORCHINSTALL="/usr/local/torch/distro/install"
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${TORCHINSTALL} -DCMAKE_INSTALL_PREFIX=${TORCHINSTALL} && make
export ORIGIN=${TORCHINSTALL}
make install
mv ${TORCHINSTALL}/lib/libpoten.so ${TORCHINSTALL}/lib/lua/5.1/.
mv ${TORCHINSTALL}/lua/poten ${TORCHINSTALL}/share/lua/5.1/
rm -rf ${TORCHINSTALL}/lua

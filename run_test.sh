export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
xvfb-run -s "-screen 0 1024x768x24" python /autodl-fs/data/relight.py "$@"

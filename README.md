
# Fazer Coppelia e Vrep funcionarem
sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev

## Executar script
nohup xvfb-run --auto-servernum --server-num=1 -s "-screen 0 1024x768x24" python3 Main.py &
conda activate dmcgen

# setting up mujoco
wget https://www.roboti.us/download/mujoco200_linux.zip
unzip mujoco200_linux
mkdir ~/.mujoco
cp -r mujoco200_linux ~/.mujoco
cp -r ~/.mujoco/mujoco200_linux ~/.mujoco/mujoco200

pip install mujoco-py

cd dmc_gen/env/dm_control
pip install -e . -q
cd ../dmc2gym
pip install -e . -q
cd ../../..
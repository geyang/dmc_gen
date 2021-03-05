from ml_logger import logger
from dmc_gen_analysis import RUN

for i in range(9):
    logger.configure(root_dir=RUN.server, prefix=f"/geyang/dmc_gen/2021/03-05/00_setup/train/01.17.36/{i}")
    logger.log_text("""
    keys:
    - Args.seed
    - Args.algorithm
    charts:
    - yKey: episode_reward/mean
      xKey: step
    - yKey: train/episode_reward/mean
      xKey: step
    """, filename=".charts.yml", overwrite=True)


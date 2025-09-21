import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common import logger
from Democracy_env import DemocracyEnv
from TimestepLogger import TimestepLogger

def enable_print_logging(log_filename):
    """
    هر چیزی که با print نمایش داده شود را علاوه بر کنسول،
    در فایل متنی log_filename هم می‌نویسد.
    """
    original_stdout = sys.stdout
    class Tee:
        def __init__(self, original, logfile):
            self.original = original
            self.logfile = logfile

        def write(self, message):
            # چاپ به کنسول
            self.original.write(message)
            # نوشتن در فایل
            self.logfile.write(message)

        def flush(self):
            self.original.flush()
            self.logfile.flush()

    # باز کردن فایل لاگ در حالت افزودن (append)
    log_file = open(log_filename, 'a', encoding='utf-8')
    sys.stdout = Tee(original_stdout, log_file)

def setup_sb3_logger():
    logger.configure(
        folder="sb3_logs",
        format_strings=["stdout", "csv", "tensorboard"]
    )

def main():
    # فعال‌سازی لاگ گرفتن از همهٔ printها
    enable_print_logging("training.txt")

    # پیکربندی لاگر داخلی SB3
    setup_sb3_logger()

    # ساخت محیط و کال‌بک
    env = DemocracyEnv()
    timestep_logger = TimestepLogger()

    # تعریف مدل PPO
    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./tensorboard/",
        ent_coef=0.05,
        learning_rate=3e-4,
    )

    try:
        print("Training Started ...")
        model.learn(
            total_timesteps=10000,
            callback=timestep_logger,
            tb_log_name="first_run",
            log_interval=4,
        )
    finally:
        timestep_logger.save_log()
        model.save("ppo_democracy4_final")
        print("Training Completed")

if __name__ == "__main__":
    main()

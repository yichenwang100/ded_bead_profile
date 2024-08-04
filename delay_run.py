import datetime
import time
import subprocess

# 设置目标日期和时间
target_datetime = datetime.datetime(year=2024,
                                    month=8,
                                    day=4,
                                    hour=4,
                                    minute=0)

def run_batch():
    print('start to run .bat file...')
    subprocess.run(['multi_run.bat'], shell=True)

def run_python_scripts():
    py_list = ["xxx.py",
               "yyy.py", ]

    for py_file in py_list:
        try:
            print(f"Started to run {py_file}")
            subprocess.run(["python", py_file], check=True)
        except subprocess.CalledProcessError:
            print(f"Failed to run {py_file} successfully.")

    print("Attempted to run all scripts.")


# 检查当前时间直到它等于或超过目标时间
while datetime.datetime.now() < target_datetime:
    remaining_time = target_datetime - datetime.datetime.now()
    # 将剩余时间转换为小时、分钟和秒
    hours, remainder = divmod(remaining_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time until execution: {hours} hours, {minutes} minutes, {seconds} seconds")
    time.sleep(60)  # 每分钟更新一次时间

# 当时间达到或超过目标时间时，执行脚本
# run_python_scripts()
run_batch()
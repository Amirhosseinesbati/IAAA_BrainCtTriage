import subprocess
import json
import os
import base64
from dotenv import load_dotenv
import sys
sys.stdout.reconfigure(encoding='utf-8')

def load_environment():
    load_dotenv()
    
    # خواندن تنظیمات پایه از .env
    config = {
        "VAST_API_KEY": os.getenv("VAST_API_KEY"),
        "DAGSHUB_TOKEN": os.getenv("DAGSHUB_USER_TOKEN"),
        "DAGSHUB_USERNAME": os.getenv("DAGSHUB_REPO_OWNER"),
        "DAGSHUB_REPO_NAME": os.getenv("DAGSHUB_REPO_NAME"),
        "DAGSHUB_TRACKING_URI": os.getenv("DAGSHUB_TRACKING_URI"),
        "GIT_REPO_URL": os.getenv("GIT_REPO_URL"),
        "GIT_BRANCH": os.getenv("GIT_BRANCH", "main"),
        "KAGGLE_USERNAME": os.getenv("KAGGLE_USERNAME", ""),
        "KAGGLE_KEY": os.getenv("KAGGLE_KEY", "")
    }
    
    # خواندن متغیرهایی که از Streamlit (deployApp.py) ارسال می‌شوند
    # اگر مستقیماً اجرا شود، مقادیر پیش‌فرض را در نظر می‌گیرد
    config["GPU_TARGET"] = os.getenv("GPU_TARGET", "RTX_3090")
    config["TARGET_PIPELINE"] = os.getenv("TARGET_PIPELINE", "all")
    config["ICH_STRATEGY"] = os.getenv("ICH_STRATEGY", "nnunet")
    config["ICH_CONFIG"] = os.getenv("ICH_CONFIG", "{}")
    
    missing_vars = [k for k, v in config.items() if not v and k not in ["KAGGLE_USERNAME", "KAGGLE_KEY"]]
    if missing_vars:
        print(f"❌ Error: Missing environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
        
    return config

def run_command(command, return_output=False, silent_error=False):
    try:
        env = os.environ.copy()
        env['PYTHONUTF8'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        env.setdefault('LC_ALL', 'C.UTF-8')
        env.setdefault('LANG', 'C.UTF-8')

        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        output_bytes = result.stdout or b''
        
        try:
            output = output_bytes.decode('utf-8')
        except Exception:
            output = output_bytes.decode('utf-8', errors='replace')

        output = output.strip()

        if result.returncode != 0 and not silent_error:
            print(f"\n🛑 COMMAND FAILED: {command}")
            print(f"--- Error Details ---\n{output}\n---------------------")
            sys.exit(1)

        return output if return_output else None
    except Exception as e:
        print(f"\n❌ Subprocess execution failed: {e}")
        sys.exit(1)

def main():
    if not os.path.exists("setup_vast.sh"):
        print("❌ Error: 'setup_vast.sh' not found in the ROOT directory!")
        sys.exit(1)

    config = load_environment()
    print(f"🔍 Searching for cheapest {config['GPU_TARGET']} for Pipeline: {config['TARGET_PIPELINE']}...")
    
    run_command(f"vastai set api-key {config['VAST_API_KEY']}", silent_error=True)

    # جستجوی سرور - حداقل 30 گیگابایت فضای دیسک برای دیتاسِت پزشکی در نظر گرفته شده
    search_cmd = f"vastai search offers \"gpu_name={config['GPU_TARGET']} num_gpus=1\" -o dph --raw"
    raw_json = run_command(search_cmd, return_output=True)
    
    try:
        offers = json.loads(raw_json)
        if not offers:
            print(f"❌ No {config['GPU_TARGET']} found! Try a different GPU.")
            sys.exit(1)
            
        instance_id = str(offers[0]['id'])
        price = offers[0]['dph_total']
        print(f"✅ Found instance! ID: {instance_id} | Price: ${price:.3f}/hour")
    except Exception as e:
        print(f"❌ Failed to parse Vast.ai output: {e}")
        sys.exit(1)

    print("🚀 Renting instance and injecting setup script...")

    # پایپ‌لاین ICH ممکنه JSON داخل کانفیگ داشته باشه
    # برای جلوگیری از خراب شدن توسط shell، ICH_CONFIG رو base64 می‌کنیم
    encoded_ich_config = base64.b64encode(
        config["ICH_CONFIG"].encode("utf-8")
    ).decode("ascii")

    # پکیج کردن تمام متغیرهای محیطی برای ارسال به سرور
    env_vars_string = (
        f"-e VAST_API_KEY={config['VAST_API_KEY']} "
        f"-e INSTANCE_ID={instance_id} "
        f"-e DAGSHUB_TOKEN={config['DAGSHUB_TOKEN']} "
        f"-e DAGSHUB_USERNAME={config['DAGSHUB_USERNAME']} "
        f"-e DAGSHUB_REPO_NAME={config['DAGSHUB_REPO_NAME']} "
        f"-e DAGSHUB_TRACKING_URI={config['DAGSHUB_TRACKING_URI']} "
        f"-e GIT_REPO_URL={config['GIT_REPO_URL']} "
        f"-e GIT_BRANCH={config['GIT_BRANCH']} "
        f"-e TARGET_PIPELINE={config['TARGET_PIPELINE']} "
        f"-e ICH_STRATEGY={config['ICH_STRATEGY']} "
        f"-e ICH_CONFIG_B64={encoded_ich_config} "
        f"-e KAGGLE_USERNAME={config['KAGGLE_USERNAME']} "
        f"-e KAGGLE_KEY={config['KAGGLE_KEY']}"
    )

    # ایمیج PyTorch رسمی + اجرای اسکریپت setup
    create_cmd = (
        f"vastai create instance {instance_id} "
        f"--image pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel  "
        f"--disk 40 "
        f"--env \"{env_vars_string}\" "
        f"--onstart setup_vast.sh "
        f"--raw"
    )

    create_output = run_command(create_cmd, return_output=True)
    
    try:
        response_json = json.loads(create_output)
        if response_json.get("error"):
            print(f"\n❌ Vast.ai Failed: {response_json.get('msg')}")
            sys.exit(1)
    except json.JSONDecodeError:
        pass
        
    print("\n🎉 The server is rented successfully.")
    print(f"Pipeline '{config['TARGET_PIPELINE']}' will run, log to DagsHub, and DESTROY the server automatically.")

if __name__ == "__main__":
    main()
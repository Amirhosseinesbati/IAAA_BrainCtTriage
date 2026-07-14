#!/bin/bash

# ==========================================
# 1. Cleanup Function (Auto-Destroy)
# ==========================================
cleanup() {
    echo "🚨 Job finished or failed. Destroying Vast.ai instance..."
    export PYTHONUTF8=1
    export PYTHONIOENCODING=utf-8
    pip install --upgrade --no-cache-dir vastai
    
    # گرفتن آیدی سرور و ارسال فرمان نابودی
    INSTANCE_ID=${VAST_CONTAINERLABEL//[!0-9]/}
    PYTHONIOENCODING=utf-8 vastai destroy instance $INSTANCE_ID -y --api-key $VAST_API_KEY
}
# فعال سازی تله: به محض اتمام اسکریپت یا بروز ارور، تابع cleanup اجرا می‌شود
#trap cleanup EXIT

echo "🚀 Starting Environment Setup on Vast.ai..."

# نصب uv و ابزارهای سیستم
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
apt-get update && apt-get install -y git awscli libgl1-mesa-glx libglib2.0-0

# کلون کردن پروژه
echo "📥 Cloning repository: $GIT_REPO_URL (Branch: $GIT_BRANCH)"
git clone -b $GIT_BRANCH $GIT_REPO_URL /workspace/project
cd /workspace/project

echo "📦 Installing dependencies with uv..."
uv sync



# نصب صریح dvc برای اطمینان از وجود آن در محیط
uv run pip install dvc

# ==========================================
# 3. دریافت داده‌های خام از طریق DVC (جدید 🌟)
# ==========================================
echo "🗄️ Authenticating DVC with DagsHub..."
# تنظیم یوزرنیم و پسورد در فایل کانفیگ محلی (local) تا در گیت ذخیره نشود
echo "⚙️ Setting up DVC..."
uv run dvc remote remove origin 2>/dev/null || true
uv run dvc remote add -d origin s3://dvc

# ساخت آدرس دقیق بر اساس متغیرهایی که deploy.py فرستاده است
uv run dvc remote modify origin endpointurl "https://dagshub.com/${DAGSHUB_USERNAME}/${DAGSHUB_REPO_NAME}.s3"


# تنظیم یوزرنیم و توکن برای دسترسی به دیتای ابری
uv run dvc remote modify origin --local access_key_id "${DAGSHUB_TOKEN}"
uv run dvc remote modify origin --local secret_access_key "${DAGSHUB_TOKEN}"


echo "📥 Pulling raw data (Data/raw) via DVC..."
uv run dvc pull -r origin

if [ -d "Data/raw" ]; then
    echo "✅ Raw data successfully pulled via DVC!"
else
    echo "❌ ERROR: Data/raw folder not found after DVC pull!"
    exit 1
fi



echo "🔗 Configuring ZenML Stack with DagsHub..."
uv run zenml init
uv run zenml integration install mlflow s3 -y --uv

# متغیرهای محیطی برای اجراهای نیتیو ابزارها (خود YOLO، nnU-Net و MLS لاگ خود را مستقیم به MLflow می‌فرستند)
export AWS_ACCESS_KEY_ID=$DAGSHUB_TOKEN
export AWS_SECRET_ACCESS_KEY=$DAGSHUB_TOKEN
export AWS_DEFAULT_REGION="us-east-1"
# این خط برای YOLO و MLflow بسیار مهم است تا مستقیم به مخزن شما وصل شوند:
export MLFLOW_S3_ENDPOINT_URL="https://dagshub.com/$DAGSHUB_USERNAME/$DAGSHUB_REPO_NAME.s3"

CLIENT_KWARGS="{\"endpoint_url\": \"https://dagshub.com/$DAGSHUB_USERNAME/$DAGSHUB_REPO_NAME.s3\", \"region_name\": \"us-east-1\"}"

# ساخت و اعمال Stack (بدون experiment tracker — هر training script مستقل لاگ می‌کند)
uv run zenml stack register vast_stack -o default -a default
uv run zenml stack set vast_stack

# متغیرهای محیطی برای هدایت MLflow همه ابزارها به DagsHub
export MLFLOW_ALLOW_FILESTORE=true
export MLFLOW_TRACKING_USERNAME=$DAGSHUB_USERNAME
export MLFLOW_TRACKING_PASSWORD=$DAGSHUB_TOKEN
export MLFLOW_TRACKING_URI=$DAGSHUB_TRACKING_URI

# ==========================================
# اجرای کدهای شما بر اساس درخواست
# ==========================================
echo "🔥 Starting Pipeline: $TARGET_PIPELINE"
# اینجا به جای pipeline قبلی، run_pipeline.py را که در چت قبل ساختیم صدا میزنیم
uv run python -m src.pipelines.run_pipeline --run $TARGET_PIPELINE

echo "🎉 Operations completed successfully. Server will self-destruct now."
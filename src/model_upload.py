# Minio에 모델 업로드 및 URL 생성
from minio import Minio


def upload_model_to_minio(model_zip_path):
    # Minio에 모델 업로드
    minio_client = Minio(
        "localhost:9000",
        access_key="miniodocker",
        secret_key="minio123",
        secure=False
    )

    # 버킷 생성
    if not minio_client.bucket_exists("shared-object"):
        minio_client.make_bucket("shared-object")

    # 모델 ZIP 파일 업로드
    minio_client.fput_object(
        "shared-object",
        f"kbert.zip",
        model_zip_path
    )

    # 모델 ZIP 파일의 URL 생성
    # 해당 방식으로 생성한 URL은 모델 등록시 에러 발생
    # 직접 미니오 콘솔에 들어가 share 버튼을 눌러 나온 Url을 사용해야함
    """
    model_zip_url = minio_client.presigned_get_object(
        "shared-object",
        f"kbert.zip",
        expires=timedelta(hours=2)
    )
    encoded_url = base64.urlsafe_b64encode(model_zip_url.encode()).decode()
    share_url = f"http://minio:9001/api/v1/download-shared-object/{encoded_url}"

    return share_url
    """

model_zip_path = "../model/kbert.zip"
upload_model_to_minio(model_zip_path)
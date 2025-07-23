export MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
export AWS_ACCESS_KEY_ID=YCAJE3Nlz8iDILW5VTYM1ihQB
export AWS_SECRET_ACCESS_KEY=YCPjvS7uwhvJpUj3bKm8X-IX4QAwBIVsvX61IL44
export S3_BUCKET_NAME=s3-student-mle-20250228-7a89d0ddfb

mlflow server \
  --backend-store-uri postgresql://mle_20250228_7a89d0ddfb:fde10607fec146789a48b0002ce9f9c4@rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net:6432/playground_mle_20250228_7a89d0ddfb \
    --default-artifact-root s3://s3-student-mle-20250228-7a89d0ddfb \
    --no-serve-artifacts
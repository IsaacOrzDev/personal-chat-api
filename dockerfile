FROM public.ecr.aws/docker/library/python:3.10.0-slim-bullseye
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.7.1 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app
ADD . .
RUN pip install -r requirements.txt

ENV PORT=8000

EXPOSE 8000
CMD ["uvicorn", "app.main:app"]
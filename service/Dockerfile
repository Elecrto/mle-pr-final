FROM python:3.11-slim
LABEL author=${AUTHOR}
COPY . ./app
WORKDIR app 
RUN pip3 install -r requirements.txt
EXPOSE ${APP_PORT}
VOLUME /models
CMD uvicorn recommendations_service:app --host  0.0.0.0 --port ${APP_PORT}

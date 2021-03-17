FROM python:3.8-buster
ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
ENV PORT 80
COPY . /app
WORKDIR /app
RUN apt update -y && apt-get install libsndfile1 -y
RUN pip install -r requirements.txt
EXPOSE 80
# CMD exec flask run --host=$FLASK_HOST_IP --port=$PORT
CMD exec waitress-serve --port $PORT main:app

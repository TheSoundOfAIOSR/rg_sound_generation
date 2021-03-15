# Audio Annotator for Web

### With Docker

```commandline
git clone \this\rep
cd \this\repo

docker build -t <image_name> .
docker run --rm -it -p 5000:5000 --name <container_name> <image_name>
```

Now you should be able to visit the app on http://127.0.0.1:5000

### Without Docker

```commandline
git clone \this\rep
cd \this\repo

virtualenv venv
venv\Scripts\activate

python -m pip install -r requirements.txt

mkdir audio_annotator\static
cp \your\wave\files audio_annotator\static

set FLASK_APP=audio_annotator
set FLASK_ENV=development

flask build-db
flask generate-spectrograms

flask run
```

Now you should be able to visit the app on http://127.0.0.1:5000

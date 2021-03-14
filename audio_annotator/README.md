# Audio Annotator for Web

Setup and run the app by following the steps below:

```
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

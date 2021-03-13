Create development environment:

```commandline
conda create --name env_name --file conda.txt
conda activate env_name
```

Run the app:

```commandline
set FLASK_APP annotator
set FLASK_ENV development
flask run
```

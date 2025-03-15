# FyledAI

Need to use python 3.10 [Hard Requirement]

Download and cache model
```shell
$> python ensure_model.py
```


Run to start a server
```shell
$> python3 app.py
```

Copy paste this curl into postman to use it

Healthcheck API
```curl
curl http://0.0.0.0:5001/api/health 
```

```curl
curl --location 'http://localhost:5001/api/process-image' \
--form 'image=@"/Users/junaid/Downloads/Aspirin.png"'
```


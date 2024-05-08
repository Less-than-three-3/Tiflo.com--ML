## Генерация кода клиента и сервера (grpc)

В папке protos
```
./generate.sh
```

P.S Для установки модуля gRPC:  <code>python -m pip install grpcio</code>

## Запуск сервера 

### Через Docker

Сборка образа
```
docker build --no-cache	-t img2seq_server_image .
```

Запуск контейнера

Перед запуском:

- указать путь до изображений 
- проверить используемую модель в конфиг файле (config/imagecap_server.yaml)

```
docker run --name img2seq_server -p 8080:8080 -v $(pwd):/image2seq -v <path to data>:/data img2seq_server_image
```

### Через venv

```
python -m venv venv
. ./venv/bin/activate
pip install -r requirements.txt
python img_server.py
```
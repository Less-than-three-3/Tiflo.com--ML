### Генерация кода клиента и сервера (grpc)

В папке protos
```
./generate.sh
```

### Запуск сервера 

Через venv

```
python -m venv venv
. ./venv/bin/activate
pip install -r requirements.txt
python img_server.py
```

Через Docker

Сборка образа
```
docker build --no-cache	-t img2seq_server_image .
```

Запуск контейнера
Перед запуском необходимо указать путь до изображений
```
docker run --name img2seq_server -p 8080:8080 -v $(pwd):/image2seq -v <path to data>:/data img2seq_server_image
```

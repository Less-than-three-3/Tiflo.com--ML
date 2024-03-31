### Генерация кода клиента и сервера (grpc)

```
cd protos

python3 -m grpc_tools.protoc -I../../image2seq/protos --python_out=. --pyi_out=. --grpc_python_out=. img2seq.proto

cd ..
```

В случае ошибки `ModuleNotFoundError: No module named 'img2seq_pb2'` заменить в файле `protos/img2seq_pb2_grpc.py` `import protos.img2seq_pb2 as img2seq__pb2` на `import protos.img2seq_pb2 as img2seq__pb2`


### Запуск сервера 

Через venv


```
python -m venv venv
. ./venv/bin/activate
pip install -r requirements.txt
python img_server.py
```

Через Docker

```
docker build --no-cache	-t img2seq_server_image .
```

```
docker run --name img2seq_server -p 8080:8080 -v $(pwd):/image2seq img2seq_server_image
```
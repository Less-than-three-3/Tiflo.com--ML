#!/bin/bash

python3 -m grpc_tools.protoc -I../protos --python_out=. --pyi_out=. --grpc_python_out=. img2seq.proto
sed  -i  's/import img2seq_pb2 as img2seq__pb2/import protos.img2seq_pb2 as img2seq__pb2/' img2seq_pb2_grpc.py 
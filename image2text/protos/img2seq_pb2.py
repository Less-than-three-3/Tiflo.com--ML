# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: img2seq.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rimg2seq.proto\x12\x02pb\"\x1b\n\x05Image\x12\x12\n\nimage_path\x18\x01 \x01(\t\"\x14\n\x04Text\x12\x0c\n\x04text\x18\x01 \x01(\t26\n\x0fImageCaptioning\x12#\n\x0cImageCaption\x12\t.pb.Image\x1a\x08.pb.TextB\x14Z\x12pkg/grpc/generatedb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'img2seq_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'Z\022pkg/grpc/generated'
  _globals['_IMAGE']._serialized_start=21
  _globals['_IMAGE']._serialized_end=48
  _globals['_TEXT']._serialized_start=50
  _globals['_TEXT']._serialized_end=70
  _globals['_IMAGECAPTIONING']._serialized_start=72
  _globals['_IMAGECAPTIONING']._serialized_end=126
# @@protoc_insertion_point(module_scope)
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: test.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\ntest.proto\x12\x02pb\"\x1b\n\x0bTextToVoice\x12\x0c\n\x04text\x18\x01 \x01(\t\"\x16\n\x05\x41udio\x12\r\n\x05\x61udio\x18\x01 \x01(\t27\n\tAIService\x12*\n\x0cVoiceTheText\x12\x0f.pb.TextToVoice\x1a\t.pb.AudioB\x14Z\x12pkg/grpc/generatedb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'test_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'Z\022pkg/grpc/generated'
  _globals['_TEXTTOVOICE']._serialized_start=18
  _globals['_TEXTTOVOICE']._serialized_end=45
  _globals['_AUDIO']._serialized_start=47
  _globals['_AUDIO']._serialized_end=69
  _globals['_AISERVICE']._serialized_start=71
  _globals['_AISERVICE']._serialized_end=126
# @@protoc_insertion_point(module_scope)

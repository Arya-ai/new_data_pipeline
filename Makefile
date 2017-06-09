# compile .proto file
datum_pb2.py: datum.proto
	protoc --python_out=. datum.proto


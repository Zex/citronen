from kafka import KafkaProducer
from kafka.errors import KafkaError
import msgpack

host = '192.168.0.101'
port = 9092
TOPIC_INPUT = 'input_x'
GRP_OUTPUT = 'predict'

producer = KafkaProducer(bootstrap_servers=[\
        '{}:{}'.format(host, port)
        ])#'broker1:1234'])

future = producer.send(TOPIC_INPUT, b'raw_bytes')

try:
    record_metadata = future.get(timeout=10)
except KafkaError as ex:
    print(ex)
    pass

print (record_metadata.topic)
print (record_metadata.partition)
print (record_metadata.offset)

producer.send(TOPIC_INPUT, key=b'foo', value=b'bar')

producer = KafkaProducer(value_serializer=msgpack.dumps)
producer.send('msgpack-topic', {
    'global_id': '8282',
    'input_x': ['sdf srsf', 'sdfe', 'sdfei'],
    })

producer = KafkaProducer(value_serializer=lambda m: json.dumps(m).encode('ascii'))
producer.send('json-topic', {'key': 'value'})

for _ in range(100):
    producer.send(TOPIC_INPUT, b'msg')

producer.flush()

producer = KafkaProducer(retries=5)

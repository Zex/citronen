from kafka import KafkaConsumer

host = '192.168.0.101'
port = 9092

TOPIC_INPUT = 'input_x'
GRP_OUTPUT = 'predict'

consumer = KafkaConsumer(TOPIC_INPUT,
                         group_id=GRP_OUTPUT,
                         bootstrap_servers=['{}:{}'.format(host, port)])#'localhost:9092'])
for message in consumer:
    print ("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
                                          message.offset, message.key,
                                          message.value))

KafkaConsumer(auto_offset_reset='earliest', enable_auto_commit=False)

KafkaConsumer(value_deserializer=lambda m: json.loads(m.decode('ascii')))

KafkaConsumer(value_deserializer=msgpack.unpackb)

KafkaConsumer(consumer_timeout_ms=1000)

consumer = KafkaConsumer()
consumer.subscribe(pattern='^awesome.*')

consumer1 = KafkaConsumer(TOPIC_INPUT,
                          group_id=GRP_OUTPUT,
                          bootstrap_servers='{}:{}'.format(host, port))
consumer2 = KafkaConsumer(TOPIC_INPUT,
                          group_id=GRP_OUTPUT,
                          bootstrap_servers='{}:{}'.format(host, port))

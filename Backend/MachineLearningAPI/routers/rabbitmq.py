import pika
import os
from config import logger, RABBITMQ_URL

def callback(ch, method, properties, body):
    logger.info(f"Received message: {body.decode()}")
    # Add your message processing logic here


def consume_from_queue(queue_name):
    connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
    channel = connection.channel()
    channel.queue_declare(queue=queue_name, durable=True)
    logger.info(f"Waiting for messages in queue: {queue_name}. To exit press CTRL+C")
    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logger.info("Stopped consuming.")
        channel.stop_consuming()
    finally:
        connection.close()

# Example usage:
# consume_from_queue('your_queue_name')

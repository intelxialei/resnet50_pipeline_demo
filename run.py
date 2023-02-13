import logging
import os
import tensorflow as tf
import numpy as np
from time import time
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes
from tensorflow.core.protobuf import rewriter_config_pb2
import intel_extension_for_tensorflow as itex

INPUTS = ''
OUTPUTS = ''

logging.basicConfig(level=logging.INFO)

def run_inference(dataset, params):

    setup(params)

    infer_config = tf.compat.v1.ConfigProto()
    infer_config.intra_op_parallelism_threads = 0
    infer_config.inter_op_parallelism_threads = 0
    infer_config.use_per_session_threads = 1
    infer_config.graph_options.rewrite_options.remapping = (
                  rewriter_config_pb2.RewriterConfig.AGGRESSIVE)

    infer_graph = create_model(params)

    input_tensor = infer_graph.get_tensor_by_name(INPUTS+':0')
    output_tensor = infer_graph.get_tensor_by_name(OUTPUTS+':0')

    infer_sess = tf.compat.v1.Session(graph=infer_graph) #, config=infer_config)

    def infer(x):
        return infer_sess.run(output_tensor, {input_tensor: x})

    def warmup():
        logging.info("warmming up for {} steps".format(params.log_warmup_steps))
        dummy_batch = np.empty((params.eval_batch_size, *params.image_size, 3))
        for i in range(params.log_warmup_steps):
            infer(dummy_batch)
    
    warmup()

    logging.info("start to inference {} samples in {} steps".format(len(dataset), dataset.chunks() ))
    total0 = time()
    if not params.async_inference:
        preprocess_time = []
        infer_time = []
        i=0
        while True:
            bt = time()        
            x = next(dataset, None)
            if x is None:
                break
            delta_time = time() - bt
            logging.info('Preprocessing %d: %.3f sec' % (i, delta_time))
            if i >= params.log_warmup_steps:
                preprocess_time.append(delta_time)
            bt = time()        
            infer(x)
            delta_time = time() - bt
            logging.info('Inference %d: %.3f sec (throughput=%.3f)' % (i, delta_time, 1.0*params.eval_batch_size/delta_time))
            if i >= params.log_warmup_steps:
                infer_time.append(delta_time)
            i +=1
        logging.info('Inference average: %.3f sec (throughput=%.3f)' % (sum(infer_time)/len(infer_time), 1.0*params.eval_batch_size/(sum(infer_time)/len(infer_time))))
    else:
        from threading import Thread
        from threading import Barrier
        from queue import Queue        
        def producer(barrier, queue, identifier):
            logging.info(f'Producer {identifier}: Running')
            while True:
                x = next(dataset, None)
                if x is None:
                    break
                queue.put(x)
            # wait for all producers to finish
            barrier.wait()
            # signal that there are no further items
            if identifier == 0:
                queue.put(None)
            logging.info(f'Producer {identifier}: Done')

        def consumer(queue,  identifier):
            # consume items
            logging.info(f'Consumer {identifier}: Running')
            i=0
            while True:
                # get a unit of work
                x = queue.get()
                # check for stop
                if x is None:
                    # add the signal back for other consumers
                    queue.put(x)
                    # stop running
                    break
                bt = time()        
                infer(x)
                delta_time = time() - bt
                logging.info('Inference %d: %d: %.3f sec' % (identifier, i, delta_time))
                i +=1
            logging.info(f'Consumer {identifier}: Done')
        # create the shared queue
        queue = Queue()
        # create the shared barrier
        n_producers = 4
        n_consumers = 1
        barrier = Barrier(n_producers)
        # start the consumer
        consumers = [Thread(target=consumer, args=(queue, i)) for i in range(n_consumers)]
        for consumer in consumers:
            consumer.start()
        # create the producers
        producers = [Thread(target=producer, args=(barrier,queue,i)) for i in range(n_producers)]
        # start the producers
        for producer in producers:
            producer.start()
        # wait for all threads to finish
        for producer in producers:
            producer.join()
        for consumer in consumers:
            consumer.join()

    '''
    mask_rcnn_model.predict(
        x=dataset.eval_fn(params.eval_batch_size),
        callbacks=list(create_callbacks(params))
    )
    '''
    total_time =time() - total0
    logging.info('Total %.3f sec' % (total_time))


def setup(params):

    # enforces that AMP is enabled using --amp and not env var
    # mainly for NGC where it is enabled by default
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

    if params.xla:
        tf.config.optimizer.set_jit(True)
        logging.info('XLA is activated')

    if params.amp:
        policy = tf.keras.mixed_precision.Policy("mixed_bfloat16")
        tf.keras.mixed_precision.set_global_policy(policy)

        # set configure for auto mixed precision.
        auto_mixed_precision_options = itex.AutoMixedPrecisionOptions()
        auto_mixed_precision_options.data_type = itex.BFLOAT16
        graph_options = itex.GraphOptions(auto_mixed_precision_options=auto_mixed_precision_options)
        # enable auto mixed precision.
        graph_options.auto_mixed_precision = itex.ON
        config = itex.ConfigProto(graph_options=graph_options)
        # set GPU backend.
        itex.set_backend('gpu', config)

        logging.info('AMP is activated')
        #exit()


def create_model(params):
    global INPUTS, OUTPUTS
    print(params)
    
    if params.version == 2:
        INPUTS = 'input_1'
        OUTPUTS = 'Identity'
    if params.version == 1:
        INPUTS = 'input_tensor'
        OUTPUTS = 'softmax_tensor'


    if params.mode =="infer":
        infer_graph = tf.Graph()
        with infer_graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(params.pb_path, 'rb') as input_file:
                input_graph_content = input_file.read()
                graph_def.ParseFromString(input_graph_content)
            output_graph = optimize_for_inference(graph_def, [INPUTS], 
                            [OUTPUTS], dtypes.float32.as_datatype_enum, False)
            tf.import_graph_def(output_graph, name='')
        return infer_graph



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import tensorflow as tf
from tensorflow import feature_column as fc
import codecs


flags = tf.app.flags
flags.DEFINE_string("model_dir", "./model_dir", "Base directory for the model.")
flags.DEFINE_string("output_model", "./model_output", "Path to the training data.")
flags.DEFINE_string("train_data", "data/samples", "Directory for storing mnist data")
flags.DEFINE_string("eval_data", "data/eval", "Path to the evaluation data.")
flags.DEFINE_string("hidden_units", "512,256,128", "Comma-separated list of number of units in each hidden layer of the NN")
flags.DEFINE_integer("train_steps", 20000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 256, "Training batch size")
flags.DEFINE_integer("shuffle_buffer_size", 10000, "dataset shuffle buffer size")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate")
flags.DEFINE_float("dropout_rate", 0.25, "Drop out rate")
flags.DEFINE_integer("num_parallel_readers", 5, "number of parallel readers for training data")
flags.DEFINE_integer("save_checkpoints_steps", 5000, "Save checkpoints every this many steps")
flags.DEFINE_string("ps_hosts", "s-xiasha-10-2-176-43.hx:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "s-xiasha-10-2-176-42.hx:2223,s-xiasha-10-2-176-44.hx:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or ps")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_boolean("run_on_cluster", False, "Whether the cluster info need to be passed in as input")

FLAGS = flags.FLAGS
my_feature_columns = []



directors_path = r"directors.json"
actors_path = r"actors.json"
directors_dict = json.load(codecs.open(directors_path, "r", "utf-8-sig"))
directors = directors_dict.keys()
print(directors)
actors_dict = json.load(codecs.open(actors_path,"r", "utf-8-sig"))
actors = actors_dict.keys()
print(actors)


def set_tfconfig_environ():
    if "TF_CLUSTER_DEF" in os.environ:
      cluster = json.loads(os.environ["TF_CLUSTER_DEF"])
      task_index = int(os.environ["TF_INDEX"])
      task_type = os.environ["TF_ROLE"]

      tf_config = dict()
      worker_num = len(cluster["worker"])
      if task_type == "ps":
        tf_config["task"] = {"index": task_index, "type": task_type}
        FLAGS.job_name = "ps"
        FLAGS.task_index = task_index
      else:
        if task_index == 0:
          tf_config["task"] = {"index": 0, "type": "chief"}
        else:
          tf_config["task"] = {"index": task_index - 1, "type": task_type}
        FLAGS.job_name = "worker"
        FLAGS.task_index = task_index

      if worker_num == 1:
        cluster["chief"] = cluster["worker"]
        del cluster["worker"]
      else:
        cluster["chief"] = [cluster["worker"][0]]
        del cluster["worker"][0]

      tf_config["cluster"] = cluster
      os.environ["TF_CONFIG"] = json.dumps(tf_config)
      print("TF_CONFIG", json.loads(os.environ["TF_CONFIG"]))

    if "INPUT_FILE_LIST" in os.environ:
      INPUT_PATH = json.loads(os.environ["INPUT_FILE_LIST"])
      if INPUT_PATH:
        print("input path:", INPUT_PATH)
        FLAGS.train_data = INPUT_PATH.get(FLAGS.train_data)
        FLAGS.eval_data = INPUT_PATH.get(FLAGS.eval_data)
      else:  # for ps
        print("load input path failed.")
        FLAGS.train_data = None
        FLAGS.eval_data = None


def parse_argument():
    if FLAGS.job_name is None or FLAGS.job_name == "":
      raise ValueError("Must specify an explicit `job_name`")
    if FLAGS.task_index is None or FLAGS.task_index == "":
      raise ValueError("Must specify an explicit `task_index`")

    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)
    os.environ["TF_ROLE"] = FLAGS.job_name
    os.environ["TF_INDEX"] = str(FLAGS.task_index)

    # Construct the cluster and start the server
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    cluster = {"worker": worker_spec, "ps": ps_spec}
    os.environ["TF_CLUSTER_DEF"] = json.dumps(cluster)


def create_feature_columns():
    age = fc.embedding_column(fc.categorical_column_with_identity("age", 10, default_value=0), 10, combiner='sqrtn')
    province = fc.embedding_column(fc.categorical_column_with_identity("province", 100, default_value=0), 10, combiner='sqrtn')
    city = fc.embedding_column(fc.categorical_column_with_identity("city", 600, default_value=0),10, combiner='sqrtn')
    citylevel = fc.embedding_column(fc.categorical_column_with_identity("citylevel", 5, default_value=0),10, combiner='sqrtn')
    devicename = fc.embedding_column(fc.categorical_column_with_identity("devicename", 200, default_value=0),200,combiner='sqrtn')


    videoid = fc.embedding_column(fc.categorical_column_with_identity("videoid", 20000, default_value=0),1000,combiner='sqrtn')
    videoscore = fc.numeric_column("videoscore", default_value=0.0)
    videoduration = fc.numeric_column("videoduration", default_value=0.0)

    videodirector = fc.embedding_column(fc.categorical_column_with_identity('videodirector', num_buckets=1000, default_value=0),100,combiner='sqrtn')
    videoactor = fc.embedding_column(fc.categorical_column_with_identity('videoactor', num_buckets=1000,  default_value=0),100,combiner='sqrtn')

    global my_feature_columns
    my_feature_columns = [age, province, city, citylevel, devicename, videoid,videoscore,videoduration,videodirector,videoactor]
    print("feature columns:", my_feature_columns)
    return my_feature_columns


def parse_exmp(serial_exmp):
    watch = fc.numeric_column("watch", default_value=0, dtype=tf.int64)
    share = fc.numeric_column("share", default_value=0, dtype=tf.int64)
    fea_columns = [watch, share]
    fea_columns += my_feature_columns
    feature_spec = tf.feature_column.make_parse_example_spec(fea_columns)
    feats = tf.parse_single_example(serial_exmp, features=feature_spec)
    watch = feats.pop('watch')
    share = feats.pop('share')
    return feats, {'ctr': tf.to_float(watch), 'cvr': tf.to_float(share)}


def train_input_fn(filenames, batch_size, shuffle_buffer_size):
    files = tf.data.Dataset.list_files(filenames)
    dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers))
    # Shuffle, repeat, and batch the examples.
    if shuffle_buffer_size > 0:
      dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_exmp, num_parallel_calls=8)
    dataset = dataset.repeat().batch(batch_size).prefetch(1)
    return dataset


def eval_input_fn(filename, batch_size):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_exmp, num_parallel_calls=8)
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.batch(batch_size)
    # Return the read end of the pipeline.
    return dataset


def build_mode(features, mode, params):
    net = fc.input_layer(features, params['feature_columns'])
    print("input shape：",net.shape)

    # Build the hidden layers, sized according to the 'hidden_units' param.
    for units in params['hidden_units']:
      net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    logits = tf.layers.dense(net, 1, activation=None)
    return logits

def multi_category_focal_loss1(y_true,y_pred,alpha, gamma=2.0):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha用于指定不同类别/标签的权重，数组大小需要与类别个数一致
    当你的数据集不同类别/标签之间存在偏斜，可以尝试适用本函数作为loss
    Usage:
     model.compile(loss=[multi_category_focal_loss1(alpha=[1,2,3,2], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    alpha = tf.transpose(tf.expand_dims(tf.constant(alpha, dtype=tf.float32),-1),perm=[1,0])
    gamma = float(gamma)
    def multi_category_focal_loss1_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -tf.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        print(tf.multiply(weight, ce).shape)
        print(alpha.shape)
        fl = tf.matmul(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)
        return loss
    return multi_category_focal_loss1_fixed(y_true,y_pred)

def my_model(features, labels, mode, params):
    epsilon = 1.e-7
    with tf.variable_scope('ctr_model'):
      ctr_logits = build_mode(features, mode, params)
    with tf.variable_scope('cvr_model'):
      cvr_logits = build_mode(features, mode, params)

    ctr_predictions = tf.sigmoid(ctr_logits+epsilon, name="CTR")
    cvr_predictions = tf.sigmoid(cvr_logits+epsilon, name="CVR")
    prop = tf.multiply(ctr_predictions, cvr_predictions, name="CTCVR")


    # predict模式的返回定义
    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
        'probabilities': prop,
        'ctr_probabilities': ctr_predictions,
        'cvr_probabilities': cvr_predictions
      }
      export_outputs = {
        'prediction': tf.estimator.export.PredictOutput(predictions)
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    y_ctr = labels['ctr']
    y_cvr = labels['cvr'] * labels['ctr']
    cvr_loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(y_cvr, prop+epsilon), name="cvr_loss")
    # cvr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_cvr, logits=cvr_logits), name="cvr_loss")
    ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_ctr, logits=ctr_logits+epsilon), name="ctr_loss")

    loss = tf.add(0.5*ctr_loss, 0.5*cvr_loss, name="ctcvr_loss")

    ctr_accuracy = tf.metrics.accuracy(labels=y_ctr, predictions=tf.to_float(tf.greater_equal(ctr_predictions, 0.5)))
    cvr_accuracy = tf.metrics.accuracy(labels=y_cvr, predictions=tf.to_float(tf.greater_equal(prop, 0.5)))
    ctr_auc = tf.metrics.auc(y_ctr, ctr_predictions)
    cvr_auc = tf.metrics.auc(y_cvr, prop)
    metrics = {'ctr_auc': ctr_auc, 'cvr_auc': cvr_auc}
    tf.summary.scalar('ctr_accuracy', ctr_accuracy[1])
    tf.summary.scalar('cvr_accuracy', cvr_accuracy[1])
    tf.summary.scalar('ctr_auc', ctr_auc[1])
    tf.summary.scalar('cvr_auc', cvr_auc[1])

    # eval模式的返回定义
    if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)



    # Create training op.
    # train模式的返回定义
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def main(unused_argv):
    set_tfconfig_environ()
    create_feature_columns()
    classifier = tf.estimator.Estimator(
      model_fn=my_model,
      params={
        'feature_columns': my_feature_columns,
        'hidden_units': FLAGS.hidden_units.split(','),
        'learning_rate': FLAGS.learning_rate,
        'dropout_rate': FLAGS.dropout_rate
      },

      config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    )

    batch_size = FLAGS.batch_size
    print("train steps:", FLAGS.train_steps, "batch_size:", batch_size)
    if isinstance(FLAGS.train_data, str) and os.path.isdir(FLAGS.train_data):
        train_files = [os.path.join(FLAGS.train_data , x) for x in os.listdir(FLAGS.train_data)] if os.path.isdir(
          FLAGS.train_data) else FLAGS.train_data
    else:
        train_files = FLAGS.train_data
    if isinstance(FLAGS.eval_data, str) and os.path.isdir(FLAGS.eval_data):
        eval_files = [os.path.join(FLAGS.eval_data , x) for x in os.listdir(FLAGS.eval_data)] if os.path.isdir(
          FLAGS.eval_data) else FLAGS.eval_data
    else:
        eval_files = FLAGS.eval_data
    shuffle_buffer_size = FLAGS.shuffle_buffer_size
    print("train_data:", train_files)
    print("eval_data:", eval_files)
    print("shuffle_buffer_size:", shuffle_buffer_size)

    train_spec = tf.estimator.TrainSpec(
      input_fn=lambda: train_input_fn(train_files, batch_size, shuffle_buffer_size),
      max_steps=FLAGS.train_steps
    )

    input_fn_for_eval = lambda: eval_input_fn(eval_files, batch_size)

    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, throttle_secs=600, steps=None)

    print("before train and evaluate")
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    print("after train and evaluate")

    # Evaluate accuracy.
    results = classifier.evaluate(input_fn=input_fn_for_eval)

    for key in sorted(results):
        print('%s: %s' % (key, results[key]))
    print("after evaluate")

    #if FLAGS.job_name == "worker" and FLAGS.task_index == 0:
    print("exporting model ...")
    feature_spec = tf.feature_column.make_parse_example_spec(my_feature_columns)
    print(feature_spec)
    # feature_spec = dict()
    # feature_spec[""] = tf.placeholder(dtype=tf.float32, shape=[None,1442],name="name")
    # serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

    def serving_input_receiver_fn2():
        # An input receiver that expects a serialized tf.Example."""
        serialized_tf_example = tf.placeholder(dtype=tf.string, name='input_example_tensor')
        receiver_tensors = {'examples': serialized_tf_example}
        features = tf.parse_example(serialized_tf_example, feature_spec)
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    print(serving_input_receiver_fn2())
    output_model = r"E:\03-code\esmm_demo\model_output"
    classifier.export_savedmodel(export_dir_base=output_model, serving_input_receiver_fn = serving_input_receiver_fn2)
    print("quit main")

    # res = classifier.predict(input_fn=input_fn_for_eval)#,checkpoint_path='./model_dir/model.ckpt-20000.data-00000-of-00001')
    # print(res)
    # for key in res:
    #     print(key)


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" in os.environ:
      print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    if FLAGS.run_on_cluster:
      parse_argument()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)

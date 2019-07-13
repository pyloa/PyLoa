import tensorflow as tf
import numpy as np


def tensorboard_scalar(writer, tag, value, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    writer.add_summary(summary, step)


def tensorboard_array(writer, tag, value, step):
    # convert to a numpy array
    values = np.asarray(value)
    # create summaries
    _mean = tf.Summary(value=[tf.Summary.Value(tag=tag+'/mean', simple_value=np.mean(values))])
    _min = tf.Summary(value=[tf.Summary.Value(tag=tag+'/min', simple_value=np.min(values))])
    _max = tf.Summary(value=[tf.Summary.Value(tag=tag+'/max', simple_value=np.max(values))])
    _stddev = tf.Summary(value=[tf.Summary.Value(tag=tag+'/stddev', simple_value=np.std(values))])
    # write summaries
    writer.add_summary(_mean, step)
    writer.add_summary(_min, step)
    writer.add_summary(_max, step)
    writer.add_summary(_stddev, step)
    writer.flush()


def tensorboard_text(writer, tag, value, step=0):
    text_tensor = tf.make_tensor_proto(value, dtype=tf.string)
    meta = tf.SummaryMetadata()
    meta.plugin_data.plugin_name = "text"
    summary = tf.Summary()
    summary.value.add(tag=tag, metadata=meta, tensor=text_tensor)
    writer.add_summary(summary)
    writer.flush()


def tensorboard_histo(writer, tag, values, step, bins=1000):
    # convert to a numpy array
    values = np.array(values)
    # create histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)
    # fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values ** 2))
    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]
    # add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)
    # create and write summary
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    writer.add_summary(summary, step)
    writer.flush()

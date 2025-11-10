import glob, tensorflow as tf
from tensorflow.train import Example

one = sorted(glob.glob("/root/repo/CDPR_Dataset/cdpr_dataset/datasets/cdpr_synth/libero_spatial_no_noops/tfrecords/*.tfrecord"))[0]
it = tf.compat.v1.io.tf_record_iterator(one)  # works in TF2 too
raw = next(it)
ex = Example(); ex.ParseFromString(raw)
for k, v in ex.features.feature.items():
    kind = v.WhichOneof("kind")  # 'bytes_list' | 'float_list' | 'int64_list'
    print(f"{k}  ->  {kind}  (len={len(getattr(v, kind).value)})")

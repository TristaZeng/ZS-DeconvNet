import argparse

from model import twostage_RCAN3D_TF1 as twostage_RCAN3D
import tensorflow as tf
import os
from keras.backend import get_session
import tempfile
import shutil

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_weights_path", type=str, default='./3D_model/1224-mitosis-Mito-cell3_560_10000.h5')
    parser.add_argument("--out_path", type=str, default='./3D_model/ZS-DeconvNet-3D-LLSM-Mito')
    parser.add_argument("--insert_xy", type=int,
                        default=8)  # pad blank edge along the input image
    parser.add_argument("--insert_d", type=int,
                        default=1)  # pad blank edge along z-axis
    parser.add_argument("--upsample_flag", type=int, default=0)  # whether the model has an upsampling function
    parser.add_argument("--NSM_flag", type=int, default=0)  # whether the model apply the NS Module
    parser.add_argument("--h", type=int, default=357)  # input image height
    parser.add_argument("--w", type=int, default=357)  # input image width
    parser.add_argument("--d", type=int, default=151)  # input image width

    args = parser.parse_args()

    load_weights_path = args.load_weights_path
    out_path = args.out_path
    insert_xy = args.insert_xy
    insert_d = args.insert_d
    upsample_flag = args.upsample_flag
    NSM_flag = args.NSM_flag

    h = args.h
    w = args.w
    d = args.d

    model = twostage_RCAN3D.RCAN3D_prun([h + 2 * insert_xy, w + 2 * insert_xy, d + 2 * insert_d, 1], NSM_flag=NSM_flag,
                                        upsample_flag=upsample_flag)
    ## Load weights
    model.load_weights(load_weights_path)

    # Save Model Bundle format
    format = 'zip'
    if out_path.endswith('.zip'):
        out_path = os.path.splitext(out_path)[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpsubdir = os.path.join(tmpdir, 'model')

        # save model
        builder = tf.saved_model.builder.SavedModelBuilder(tmpsubdir)
        # use name 'input'/'output' if there's just a single input/output layer
        inputs = dict(zip(model.input_names, model.inputs)) if len(model.inputs) > 1 else dict(input=model.input)
        outputs = dict(zip(model.output_names, model.outputs)) if len(model.outputs) > 1 else dict(output=model.output)
        signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs=inputs, outputs=outputs)
        signature_def_map = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
        builder.add_meta_graph_and_variables(get_session(), [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map=signature_def_map)
        builder.save()

        shutil.make_archive(out_path, format, tmpsubdir)

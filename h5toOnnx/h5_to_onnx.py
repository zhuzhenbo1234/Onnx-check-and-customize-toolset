import keras2onnx
from keras.models import load_model
from absl import flags, app
from absl.flags import FLAGS
import os

flags.DEFINE_string('model', 'model.h5', 'path to input .h5 file')
flags.DEFINE_string('out', 'model.onnx',
                    'path to output .onnx file')

def main(_argv):
    print("Loading", FLAGS.model)
    if not os.path.exists(FLAGS.model):
        print("h5 model not found at path: {}\nUse the --model flag to specify a path to your h5 model.".format(FLAGS.model))
        return None
    model = load_model(FLAGS.model)
    print("Converting to ONNX...")
    onnx_model = keras2onnx.convert_keras(model)
    keras2onnx.save_model(onnx_model, FLAGS.out)
    print("Success. Output at:", FLAGS.out)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
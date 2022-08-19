import onnxruntime as rt
import onnx

# Preprocessing: load the ONNX model
model_path = 'models/squeezenet1.1-7.onnx'
#model_path = 'models/squeezenet1.1-7.onnx'
onnx_model = onnx.load(model_path)

#print('The model is:\n{}'.format(onnx_model))

# Check the model
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print('The model is invalid: %s' % e)
else:
    print('The model is valid!')


print("onnx.__version__ =>" + onnx.__version__)
print("rt.__version__ =>" + rt.__version__)

#sess = rt.InferenceSession(model_path)
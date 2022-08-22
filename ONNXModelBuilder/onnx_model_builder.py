### a tool for build onnx model ###
import onnx
import numpy as np
# from onnx import version_converter
# from onnx import numpy_helper


class ConvAttrs:
    def __init__(self, Ci, Co, kernel, strides=[1, 1],
                 pads=[0, 0, 0, 0], dilations=[1, 1],
                 group=1, with_bias=True):
        self.Ci = Ci
        self.Co = Co
        self.kernel = kernel
        self.strides = strides
        self.dilations = dilations
        self.pads = pads
        self.group = group
        self.with_bias = with_bias


def create_conv_node(id, input_names, attrs: ConvAttrs):
    output_name = 'conv_%s' % (id)
    w_name = 'W_%s' % (id)
    b_name = 'B_%s' % (id)
    kernel = attrs.kernel
    kernel_shape = [attrs.Co, attrs.Ci] + kernel

    values = []

    WeightVals = np.random.normal(
        size=kernel_shape, loc=0.0, scale=0.1).astype("float32")

    W = onnx.helper.make_tensor(
        name=w_name,
        data_type=onnx.TensorProto.FLOAT,
        dims=WeightVals.shape,
        vals=WeightVals.flatten(),
    )
    values.append(W)
    input_names.append(w_name)

    if attrs.with_bias:
        B = onnx.helper.make_tensor(
            name=b_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=[attrs.Co],
            vals=np.random.normal(
                size=attrs.Co, loc=0.0, scale=0.1).astype("float32")
        )
        values.append(B)
        input_names.append(b_name)

    node = onnx.helper.make_node(
        'Conv',
        inputs=input_names,
        outputs=[output_name],
        kernel_shape=kernel,
        dilations=attrs.dilations,
        group=attrs.group,
        strides=attrs.strides,
        pads=attrs.pads,
    )

    return [node], values, [output_name]


class GemmAttrs:
    def __init__(self, Ci, Co, with_bias=False,
                 alpha=None, beta=None, transA=None, transB=None):
        self.Ci = Ci
        self.Co = Co
        self.with_bias = with_bias
        self.alpha = alpha
        self.beta = beta
        self.transA = transA
        self.transB = transB


def create_gemm_node(id, input_names, attrs: GemmAttrs):
    output_name = 'gemm_%s' % (id)
    b_name = 'B_%s' % (id)
    c_name = 'C_%s' % (id)
    b_shape = [attrs.Ci, attrs.Co]
    if attrs.transB == 1:
        b_shape = [attrs.Co, attrs.Ci]

    values = []

    WeightVals = np.random.normal(
        size=b_shape, loc=0.0, scale=0.1).astype("float32")

    B = onnx.helper.make_tensor(
        name=b_name,
        data_type=onnx.TensorProto.FLOAT,
        dims=WeightVals.shape,
        vals=WeightVals.flatten(),
    )

    values.append(B)
    input_names.append(b_name)

    if attrs.with_bias:
        C = onnx.helper.make_tensor(
            name=c_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=[attrs.Co],
            vals=np.random.normal(
                size=attrs.Co, loc=0.0, scale=0.1).astype("float32")
        )
        values.append(C)
        input_names.append(c_name)

    gemm_param = {}
    if attrs.alpha is not None:
        gemm_param['alpha'] = attrs.alpha
    if attrs.beta is not None:
        gemm_param['beta'] = attrs.beta
    if attrs.transA is not None:
        gemm_param['transA'] = attrs.transA
    if attrs.transB is not None:
        gemm_param['transB'] = attrs.transB

    node = onnx.helper.make_node(
        'Gemm', input_names, [output_name], **gemm_param)

    return [node], values, [output_name]


class FlattenAttrs:
    def __init__(self, axis):
        self.axis = axis


def create_flatten_node(id, input_names, attrs: FlattenAttrs):
    output_name = 'flatten_%s' % (id)
    node = onnx.helper.make_node(
        'Flatten',
        inputs=input_names,
        outputs=[output_name],
        axis=attrs.axis,
    )

    return [node], [], [output_name]


class PoolAttrs:
    def __init__(self, kernel, strides=[1, 1], pads=[0, 0, 0, 0]):
        self.kernel = kernel
        self.strides = strides
        self.pads = pads


def create_maxpool_node(id, input_names, attrs: PoolAttrs):
    output_name = 'maxpool_%s' % (id)
    node = onnx.helper.make_node(
        'MaxPool',
        inputs=input_names,
        outputs=[output_name],
        kernel_shape=attrs.kernel,
        strides=attrs.strides,
        pads=attrs.pads,
    )

    return [node], [], [output_name]


class TransposeAttrs:
    def __init__(self, perm=[0, 3, 1, 2]):
        self.perm = perm


def create_transpose_node(id, input_names, attrs):
    output_name = 'transpose_%s' % (id)
    node = onnx.helper.make_node(
        'Transpose',
        inputs=input_names,
        outputs=[output_name],
        perm=attrs.perm)

    return [node], [], [output_name]


class CastAttrs:
    def __init__(self, type=onnx.TensorProto.FLOAT16):
        self.type = type


def create_cast_node(id, input_names, attrs: CastAttrs):
    output_name = 'cast_%s' % (id)
    node = onnx.helper.make_node(
        'Cast',
        inputs=input_names,
        outputs=[output_name],
        to=int(attrs.type),
    )

    return [node], [], [output_name]


def create_relu_node(id, input_names, attrs=None):
    output_name = 'relu_%s' % (id)
    node = onnx.helper.make_node(
        'Relu',
        inputs=input_names,
        outputs=[output_name],
    )

    return [node], [], [output_name]


def create_globalaveragepool_node(id, input_names, attrs=None):
    output_name = 'globalaveragepool_%s' % (id)
    node = onnx.helper.make_node(
        'GlobalAveragePool',
        inputs=input_names,
        outputs=[output_name],
    )

    return [node], [], [output_name]


def create_add_node(id, input_names, attrs=None):
    output_name = 'add_%s' % (id)
    node = onnx.helper.make_node(
        'Add',
        inputs=input_names,
        outputs=[output_name],
    )

    return [node], [], [output_name]


class ONNXModelBuilder:
    def __init__(self, id=0):
        self.input_tensor_list = []
        self.output_tensor_list = []
        self.node_list = []
        self.var_list = []
        self.id = id
        return

    def add_input_tensor(self, name, shape, type=onnx.TensorProto.FLOAT):
        var = onnx.helper.make_tensor_value_info(name, type, shape)
        self.input_tensor_list.append(var)
        return

    def add_output_tensor(self, name, shape=None, type=onnx.TensorProto.FLOAT):
        var = onnx.helper.make_tensor_value_info(name, type, shape)
        self.output_tensor_list.append(var)
        return

    def add_node(self, create_node_fuction, input_names, attrs=None):
        nodes, vars, output_names = create_node_fuction(
            str(self.id), input_names, attrs)
        self.node_list += nodes
        self.var_list += vars
        self.id += 1
        return output_names

    def create_model(self, model_file_name, opset=13):
        graph = onnx.helper.make_graph(
            self.node_list,
            'test_model',
            self.input_tensor_list,
            self.output_tensor_list,
        )
        for var in self.var_list:
            graph.initializer.append(var)

        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        model.opset_import[0].version = opset
        onnx.checker.check_model(model)
        # model.ir_version = 7
        onnx.save(model, model_file_name)
        print("[INFO] save model: %s" % (model_file_name))
        return


def create_node_test(input_shapes, create_node_function, attrs=None,
                     model_save_path='test-node-op13.onnx'):
    builder = ONNXModelBuilder()

    input_names = []
    for i in range(len(input_shapes)):
        input_names.append('input_%d' % (i))

    ### create node
    node_output_names = builder.add_node(
        create_node_function, input_names, attrs)

    ### create input & output tensor
    for i in range(len(input_shapes)):
        builder.add_input_tensor(
            input_names[i], input_shapes[i], type=onnx.TensorProto.FLOAT)

    for output_name in node_output_names:
        builder.add_output_tensor(
            output_name, type=onnx.TensorProto.FLOAT)

    ### create model
    builder.create_model(model_save_path)


def create_multi_node_model_test():
    #   A      B
    #   |      |
    # conv0 conv1
    #    \  /  |
    #     add  |
    #      |   |
    #    out0 out1
    input_names = ['A', 'B']
    input_shapes = [[1, 32, 16, 16], [1, 64, 8, 8]]

    builder = ONNXModelBuilder()

    ### create node
    conv0_outputs_names = builder.add_node(
        create_conv_node, [input_names[0]], ConvAttrs(
            32, 64, kernel=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
    )

    conv1_outputs_names = builder.add_node(
        create_conv_node, [input_names[1]], ConvAttrs(
            64, 64, kernel=[1, 1], strides=[1, 1], with_bias=False)
    )

    add_outputs_names = builder.add_node(
        create_add_node, [conv0_outputs_names[0], conv1_outputs_names[0]])

    output_names = [conv1_outputs_names[0], add_outputs_names[0]]

    ### create input & output tensor
    for i in range(len(input_shapes)):
        builder.add_input_tensor(
            input_names[i], input_shapes[i], type=onnx.TensorProto.FLOAT)

    for output_name in output_names:
        builder.add_output_tensor(
            output_name, type=onnx.TensorProto.FLOAT)

    ### create model
    builder.create_model('test-multi-node-model-op13.onnx', opset=13)


if __name__ == '__main__':
    print('onnx version:', onnx.__version__)
    np.random.seed(0)

    create_node_test([[1, 64, 56, 56]], create_conv_node,
                     ConvAttrs(64, 32, kernel=[3, 3], strides=[1, 1],
                               pads=[1, 1, 1, 1], with_bias=True),
                     'test-conv-node-op13.onnx')

    create_node_test([[1, 56, 56, 3]], create_transpose_node,
                     TransposeAttrs(perm=[0, 3, 1, 2]),
                     'test-transpose-node-op13.onnx')

    create_multi_node_model_test()

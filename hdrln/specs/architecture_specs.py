class ConvLayer():
    """
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride,
                    padding):
        """
        """
        self.input_channels = input_channels
        self.out_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

class ArchitectureSpecs():
    """
    """
    def __init__(self, ):
        """
        """
        pass

# 1. Convolution layers specs

INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 32
KERNEL_1 = 13
STRIDE_1 = 1
PADDING_1 = 0
conv_layer_1 = ConvLayer(
                        input_channels=INPUT_CHANNELS_1,
                        output_channels=OUTPUT_CHANNELS_1
                        kernel_size=KERNEL_1,
                        stride=STRIDE_1,
                        padding=PADDING_1
                        )


conv_specs = [
                conv_layer_1,
                conv_layer_2,
                conv_layer_3
]

architecture_specs = {
                        "CONV_SPECS": conv_specs,

}

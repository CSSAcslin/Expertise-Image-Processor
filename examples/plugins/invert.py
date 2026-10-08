import numpy as np


class Invert:
    def prepare(self, parameters, context):
        self.reference = parameters["reference"]
        context.log("示例插件已准备；原始输入不会被改写")

    def process(self, pixels, context):
        context.check_cancelled()
        return (self.reference - pixels.astype(np.float32)).astype(np.float32)

    def close(self):
        pass

import tensorflow as tf



class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filters, strides):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        if strides != 1:
            self.convdown = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=strides)
            self.bndown = tf.keras.layers.BatchNormalization()
        self.strides = strides

    def call(self, inp, training=False):
        x1 = self.conv1(inp)
        x1 = self.bn1(x1, training=training)
        x1 = tf.nn.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1, training=training)

        if self.strides != 1:
            x2 = self.convdown(inp)
            x2 = self.bndown(x2, training=training)
        else:
            x2 = inp

        x = tf.keras.layers.add([x1, x2])
        x = tf.nn.relu(x)
        return x


# ResNet with BasicBlock (adapted to CIFAR-10)
class BasicResNet(tf.keras.Model):

    def __init__(self, layer_blocks):
        super(BasicResNet, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.blocks = []
        self.blocks.append(BasicBlock(filters=64, strides=1))
        for _ in range(layer_blocks[0] - 1):
            self.blocks.append(BasicBlock(filters=64, strides=1))
        
        self.blocks.append(BasicBlock(filters=128, strides=2))
        for _ in range(layer_blocks[1] - 1):
            self.blocks.append(BasicBlock(filters=128, strides=1))
        
        self.blocks.append(BasicBlock(filters=256, strides=2))
        for _ in range(layer_blocks[2] - 1):
            self.blocks.append(BasicBlock(filters=256, strides=1))
        
        self.blocks.append(BasicBlock(filters=512, strides=2))
        for _ in range(layer_blocks[3] - 1):
            self.blocks.append(BasicBlock(filters=512, strides=1))

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, inp, training=False):
        x = self.conv1(inp)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.avgpool(x)
        return x


def ResNet18():
    return BasicResNet(layer_blocks=[2, 2, 2, 2])


def ResNet34():
    return BasicResNet(layer_blocks=[3, 4, 6, 3])


# 512 (h) -> 256 -> 128 (z)
class ProjectionHead(tf.keras.Model):

    def __init__(self):
        super(ProjectionHead, self).__init__()
        self.fc1 = tf.keras.layers.Dense(units=256)
        self.bn = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(units=128)

    def call(self, inp, training=False):
        x = self.fc1(inp)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        return x


# 512 (h) -> 10 (s)
class ClassificationHead(tf.keras.Model):

    def __init__(self):
        super(ClassificationHead, self).__init__()
        self.fc = tf.keras.layers.Dense(units=10)

    def call(self, inp):
        x = self.fc(inp)
        return x

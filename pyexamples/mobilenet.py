import sys
sys.path.append('../')
from pycore.tikzeng import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    
    # Input image
    to_input('../examples/Mobilenet/kitten_224x224.jpg'),
    
    # Initial Conv Layer
    to_Conv("conv1", 224, 224, offset="(0,0,0)", to="(0,0,0)", height=40, depth=40, width=4, caption="Conv1 (3x3)" ),
    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)", height=30, depth=30, width=4, opacity=0.5),
    
    # Depthwise Separable Convolution Blocks
    to_Conv("dconv2", 112, 112, offset="(1,0,0)", to="(pool1-east)", height=30, depth=30, width=2, caption="DW+PW Conv" ),
    to_Conv("pconv2", 112, 112, offset="(0,0,0)", to="(dconv2-east)", height=30, depth=30, width=5, caption="" ),
    to_connection("pool1", "dconv2"),
    
    to_Conv("dconv3", 56, 56, offset="(1,0,0)", to="(pconv2-east)", height=20, depth=20, width=2, caption="DW+PW Conv" ),
    to_Conv("pconv3", 56, 56, offset="(0,0,0)", to="(dconv3-east)", height=20, depth=20, width=5, caption="" ),
    to_connection("pconv2", "dconv3"),
    
    to_Conv("dconv4", 56, 56, offset="(1,0,0)", to="(pconv3-east)", height=20, depth=20, width=2, caption="DW+PW Conv" ),
    to_Conv("pconv4", 56, 56, offset="(0,0,0)", to="(dconv4-east)", height=20, depth=20, width=5, caption="" ),
    to_connection("pconv3", "dconv4"),
    
    to_Conv("dconv5", 28, 28, offset="(1,0,0)", to="(pconv4-east)", height=15, depth=15, width=2, caption="DW+PW Conv" ),
    to_Conv("pconv5", 28, 28, offset="(0,0,0)", to="(dconv5-east)", height=15, depth=15, width=5, caption="" ),
    to_connection("pconv4", "dconv5"),
    
    to_Conv("dconv6", 28, 28, offset="(1,0,0)", to="(pconv5-east)", height=15, depth=15, width=2, caption="DW+PW Conv" ),
    to_Conv("pconv6", 28, 28, offset="(0,0,0)", to="(dconv6-east)", height=15, depth=15, width=5, caption="" ),
    to_connection("pconv5", "dconv6"),
    
    to_Conv("dconv7", 14, 14, offset="(1,0,0)", to="(pconv6-east)", height=10, depth=10, width=2, caption="DW+PW Conv" ),
    to_Conv("pconv7", 14, 14, offset="(0,0,0)", to="(dconv7-east)", height=10, depth=10, width=5, caption="" ),
    to_connection("pconv6", "dconv7"),
    
    to_Conv("dconv8", 14, 14, offset="(1,0,0)", to="(pconv7-east)", height=10, depth=10, width=2, caption="DW+PW Conv" ),
    to_Conv("pconv8", 14, 14, offset="(0,0,0)", to="(dconv8-east)", height=10, depth=10, width=5, caption="" ),
    to_connection("pconv7", "dconv8"),
    
    to_Conv("dconv9", 14, 14, offset="(1,0,0)", to="(pconv8-east)", height=10, depth=10, width=2, caption="DW+PW Conv" ),
    to_Conv("pconv9", 14, 14, offset="(0,0,0)", to="(dconv9-east)", height=10, depth=10, width=5, caption="" ),
    to_connection("pconv8", "dconv9"),
    
    to_Conv("dconv10", 7, 7, offset="(1,0,0)", to="(pconv9-east)", height=7, depth=7, width=2, caption="DW+PW Conv" ),
    to_Conv("pconv10", 7, 7, offset="(0,0,0)", to="(dconv10-east)", height=7, depth=7, width=5, caption="" ),
    to_connection("pconv9", "dconv10"),
    
    to_Conv("dconv11", 7, 7, offset="(1,0,0)", to="(pconv10-east)", height=7, depth=7, width=2, caption="DW+PW Conv" ),
    to_Conv("pconv11", 7, 7, offset="(0,0,0)", to="(dconv11-east)", height=7, depth=7, width=5, caption="" ),
    to_connection("pconv10", "dconv11"),
    
    to_Pool("global_avg_pool", offset="(0,0,0)", to="(pconv11-east)", height=1, depth=1, width=5, caption="GAP"),
    to_Conv("fc", 1, 1, offset="(1,0,0)", to="(global_avg_pool-east)", height=1, depth=1, width=8, caption="FC"),
    to_SoftMax("softmax", 10, offset="(0.5,0,0)", to="(fc-east)", caption="Softmax"),
    to_connection("global_avg_pool", "fc"),
    to_connection("fc", "softmax"),
    
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()


import moderngl
import numpy as np


def div_round_up(x, y):
    return int((x + y - 1) / y)

NUM_ELEMENTS = 5000

ctx = moderngl.create_context(standalone=True)

elements_in = np.random.randint(100000, size=(NUM_ELEMENTS*2)).astype('uint32')
elements_in[NUM_ELEMENTS:] = elements_in[:NUM_ELEMENTS]
elements_in_buffer = ctx.buffer(elements_in)
elements_in_buffer.bind_to_storage_buffer(2)

indices = np.arange((2*NUM_ELEMENTS)).astype('uint32')
indices[NUM_ELEMENTS:] = indices[:NUM_ELEMENTS]
indices_buffer = ctx.buffer(indices)
indices_buffer.bind_to_storage_buffer(1)

source = open(
    '/home/liu/workspace/simple_gaussian_splatting/viewer/shaders/radix_sort.glsl', 'r').read()

compute_shader = ctx.compute_shader(source)

compute_shader['g_num_elements'] = NUM_ELEMENTS

compute_shader.run(group_x=div_round_up(NUM_ELEMENTS, 256))
indices = np.frombuffer(indices_buffer.read(), dtype='uint32')[:NUM_ELEMENTS]
sorted_in = np.frombuffer(elements_in_buffer.read(), dtype='uint32')[:NUM_ELEMENTS]
print("sorted data:", elements_in[indices])
np.arange(NUM_ELEMENTS) == np.sort(indices)
print(np.all(np.arange(NUM_ELEMENTS) == np.sort(indices)))
elements_in_buffer.clear()

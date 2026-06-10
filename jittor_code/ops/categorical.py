import jittor as jt


def categorical_sample(probs):
    shape = probs.shape
    flat = probs.reshape((-1, shape[-1]))
    uniforms = jt.rand((flat.shape[0],))

    sampled = jt.code(
        (flat.shape[0],),
        "int32",
        [flat, uniforms],
        cpu_src="""
            @alias(probs, in0)
            @alias(uniforms, in1)
            @alias(sampled, out)
            for (int i = 0; i < probs_shape0; ++i) {
                double total = 0.0;
                for (int j = 0; j < probs_shape1; ++j)
                    total += (double)@probs(i, j);

                double threshold = (double)@uniforms(i) * total;
                double cdf = 0.0;
                int idx = probs_shape1 - 1;
                for (int j = 0; j < probs_shape1; ++j) {
                    cdf += (double)@probs(i, j);
                    if (threshold <= cdf) {
                        idx = j;
                        break;
                    }
                }
                @sampled(i) = idx;
            }
        """,
        cuda_src="""
            __global__ static void categorical_sample_kernel(@ARGS_DEF) {
                @PRECALC
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                int stride = blockDim.x * gridDim.x;
                for (; i < in0_shape0; i += stride) {
                    double total = 0.0;
                    for (int j = 0; j < in0_shape1; ++j)
                        total += (double)@in0(i, j);

                    double threshold = (double)@in1(i) * total;
                    double cdf = 0.0;
                    int idx = in0_shape1 - 1;
                    for (int j = 0; j < in0_shape1; ++j) {
                        cdf += (double)@in0(i, j);
                        if (threshold <= cdf) {
                            idx = j;
                            break;
                        }
                    }
                    @out(i) = idx;
                }
            }
            int block_size = 256;
            int grid_size = (in0_shape0 + block_size - 1) / block_size;
            categorical_sample_kernel<<<grid_size, block_size>>>(@ARGS);
        """,
    )
    return sampled.reshape(shape[:-1])

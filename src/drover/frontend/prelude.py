"""Minimal CUDA declarations so clang can resolve builtins during a C++ parse.

Without this, threadIdx/__syncthreads/etc. parse as undeclared identifiers and
never resolve to CALL_EXPR or MEMBER_REF nodes, which silently degrades every
downstream structural query.
"""

PRELUDE = """
struct uint3 { unsigned x, y, z; };
struct dim3  { unsigned x, y, z; dim3(unsigned = 1, unsigned = 1, unsigned = 1); };
extern uint3 threadIdx;
extern uint3 blockIdx;
extern dim3  blockDim;
extern dim3  gridDim;
extern int   warpSize;

void __syncthreads();
void __threadfence();
void __threadfence_block();

float  fmaxf(float, float);
float  fminf(float, float);
float  expf(float);
float  logf(float);
float  sqrtf(float);
float  rsqrtf(float);
float  fabsf(float);

int    __shfl_down_sync(unsigned, int, unsigned, int = 32);
float  __shfl_down_sync(unsigned, float, unsigned, int = 32);
int    __shfl_xor_sync(unsigned, int, int, int = 32);
float  __shfl_xor_sync(unsigned, float, int, int = 32);
int    __shfl_up_sync(unsigned, int, unsigned, int = 32);
unsigned __ballot_sync(unsigned, int);

int    atomicAdd(int*, int);
float  atomicAdd(float*, float);
int    atomicMax(int*, int);
int    atomicCAS(int*, int, int);
"""

# Number of lines the prelude occupies, so reported line numbers can be
# corrected back to the user's original file.
PRELUDE_LINES = PRELUDE.count("\n")

SHUFFLE_NAMES = frozenset(
    {
        "__shfl_down_sync",
        "__shfl_xor_sync",
        "__shfl_up_sync",
        "__shfl_sync",
        "__ballot_sync",
    }
)

ATOMIC_NAMES = frozenset(
    {
        "atomicAdd",
        "atomicSub",
        "atomicMax",
        "atomicMin",
        "atomicExch",
        "atomicCAS",
        "atomicAnd",
        "atomicOr",
        "atomicXor",
    }
)

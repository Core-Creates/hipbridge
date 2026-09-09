"""Minimal CUDA declarations so clang can resolve builtins during a C++ parse.

Without this, threadIdx/__syncthreads/etc. parse as undeclared identifiers and
never resolve to CALL_EXPR or MEMBER_REF nodes, which silently degrades every
downstream structural query.

This module also owns the *vocabulary*: the sets of spellings that mean "this
kernel exponentiates", "takes a maximum", "normalises". Those sets were
previously written out twice, in `frontend/parser.py` and in
`verify/substitutions.py`, character for character, in two layers with no shared
definition. Worse, neither was reconciled against the declarations below, and
`facts.calls` can only ever contain names clang resolved. Thirteen of the
seventeen spellings the recogniser tested for were undeclarable, so they never
appeared, and an ordinary fast-math softmax written with `__expf` produced
`calls=[]` and was refused with "no recognizer claimed this kernel".

So the declarations are generated from the vocabulary rather than maintained
beside it, and a test asserts the containment. Adding a spelling to a set below
is the whole change; it cannot be added to one half and forgotten in the other.
"""

from __future__ import annotations

# --- the vocabulary --------------------------------------------------------
#
# A recognizer keys on these. Every name here MUST be declarable, which is what
# generating the declarations below from these same sets guarantees.

# Exponentiation, which is what separates a softmax from a plain row sum.
EXP_NAMES = frozenset({"exp", "expf", "__expf", "exp2f", "exp2", "hexp"})

# The stability pass a competent softmax runs before exponentiating.
MAX_NAMES = frozenset({"max", "fmax", "fmaxf", "fmaxf16", "hmax", "__hmax"})

# Where a normalisation keeps its epsilon: the reciprocal-square-root family.
NORMALISING = frozenset({"rsqrt", "rsqrtf", "sqrt", "sqrtf", "hrsqrt", "__frsqrt_rn"})

# Not vocabulary. No rule keys on these, but a kernel that calls one still has to
# parse, and an unresolved call costs the surrounding structural facts too.
_ALSO_UNARY = frozenset({"log", "logf", "__logf", "fabs", "fabsf", "tanh", "tanhf", "erff"})
_ALSO_BINARY = frozenset({"min", "fmin", "fminf", "hmin", "pow", "powf", "fmodf"})

UNARY_MATH = EXP_NAMES | NORMALISING | _ALSO_UNARY
BINARY_MATH = MAX_NAMES | _ALSO_BINARY


def _math_declarations() -> str:
    """One declaration per vocabulary name, as a template.

    Templates rather than overload sets on purpose. `float f(float)` next to
    `double f(double)` is an ambiguity waiting for the first half-precision
    argument that converts to both, and resolving that ambiguity is exactly the
    silent-degradation failure this module exists to prevent. A single template
    accepts whatever the kernel passes and still yields a CALL_EXPR whose
    spelling is the name we asked about, which is all the frontend reads.
    """
    lines = [f"template <class T> T {name}(T);" for name in sorted(UNARY_MATH)]
    lines += [f"template <class T> T {name}(T, T);" for name in sorted(BINARY_MATH)]
    return "\n".join(lines)


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

# Structural declarations, which have real signatures rather than a shape, and
# so are written out rather than generated.
_STRUCTURAL = """
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

int    __shfl_down_sync(unsigned, int, unsigned, int = 32);
float  __shfl_down_sync(unsigned, float, unsigned, int = 32);
int    __shfl_xor_sync(unsigned, int, int, int = 32);
float  __shfl_xor_sync(unsigned, float, int, int = 32);
int    __shfl_up_sync(unsigned, int, unsigned, int = 32);
int    __shfl_sync(unsigned, int, int, int = 32);
float  __shfl_sync(unsigned, float, int, int = 32);
unsigned __ballot_sync(unsigned, int);

int    atomicAdd(int*, int);
float  atomicAdd(float*, float);
int    atomicSub(int*, int);
int    atomicMax(int*, int);
int    atomicMin(int*, int);
int    atomicExch(int*, int);
float  atomicExch(float*, float);
int    atomicCAS(int*, int, int);
int    atomicAnd(int*, int);
int    atomicOr(int*, int);
int    atomicXor(int*, int);
"""

PRELUDE = _STRUCTURAL + _math_declarations() + "\n"

# Number of lines the prelude occupies, so reported line numbers can be
# corrected back to the user's original file.
PRELUDE_LINES = PRELUDE.count("\n")


def declared_names() -> frozenset[str]:
    """Every function name the prelude declares. The test target for containment."""
    import re

    return frozenset(re.findall(r"\b(\w+)\s*\(", PRELUDE)) - {"dim3"}

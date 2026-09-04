"""libclang-backed CUDA frontend.

Parses a .cu translation unit and extracts structural facts per __global__
kernel. Everything reported here comes from the AST. Nothing is inferred from
source text.
"""

from __future__ import annotations

from pathlib import Path

import clang.cindex as ci

from drover.frontend.ir import KernelFacts, Param, SharedBuffer
from drover.frontend.prelude import ATOMIC_NAMES, PRELUDE, SHUFFLE_NAMES

# __global__/__device__ are attribute keywords clang only accepts under -x cuda,
# which additionally wants the CUDA SDK headers. Parsing as C++ with the
# attributes defined away plus our own prelude keeps the frontend dependency-free.
_CLANG_ARGS = [
    "-x",
    "c++",
    "-std=c++17",
    "-ferror-limit=0",
    "-D__global__=",
    "-D__device__=",
    "-D__host__=",
    "-D__forceinline__=inline",
    "-D__restrict__=",
    "-D__launch_bounds__(...)=",
    "-D__shared__=static",  # models block-scope shared storage as static
]


class ParseError(RuntimeError):
    pass


def _walk(node):
    yield node
    for child in node.get_children():
        yield from _walk(child)


def _tokens(node) -> list[str]:
    return [t.spelling for t in node.get_tokens()]


def _to_param(cursor) -> Param:
    t = cursor.type
    spelling = t.spelling
    is_ptr = t.kind == ci.TypeKind.POINTER
    is_const = is_ptr and t.get_pointee().is_const_qualified()
    return Param(name=cursor.spelling, type=spelling, is_pointer=is_ptr, is_const=is_const)


def _kernel_facts(fn) -> KernelFacts:
    nodes = list(_walk(fn))

    shared = [
        SharedBuffer(
            name=n.spelling,
            type=n.type.spelling,
            size_bytes=n.type.get_size() if n.type.get_size() > 0 else None,
        )
        for n in nodes
        if n.kind == ci.CursorKind.VAR_DECL and n.storage_class == ci.StorageClass.STATIC
    ]
    shared_names = {s.name for s in shared}

    calls = [n for n in nodes if n.kind == ci.CursorKind.CALL_EXPR]
    barriers = sum(1 for c in calls if c.spelling == "__syncthreads")
    shuffles = sorted({c.spelling for c in calls if c.spelling in SHUFFLE_NAMES})
    atomics = sorted({c.spelling for c in calls if c.spelling in ATOMIC_NAMES})

    loops = [n for n in nodes if n.kind in (ci.CursorKind.FOR_STMT, ci.CursorKind.WHILE_STMT)]
    halving = any("/=" in _tokens(loop) or ">>=" in _tokens(loop) for loop in loops)

    compound = [n for n in nodes if n.kind == ci.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR]
    shared_acc = sum(1 for n in compound if any(t in shared_names for t in _tokens(n)))
    scalar_acc = len(compound) - shared_acc

    refs = {n.spelling for n in nodes if n.kind == ci.CursorKind.DECL_REF_EXPR}

    return KernelFacts(
        name=fn.spelling,
        params=[_to_param(a) for a in fn.get_arguments()],
        shared=shared,
        barriers=barriers,
        loops=len(loops),
        has_halving_stride=halving,
        shared_accumulations=shared_acc,
        scalar_accumulations=scalar_acc,
        shuffle_intrinsics=shuffles,
        atomics=atomics,
        uses_block_index="blockIdx" in refs,
        uses_thread_index="threadIdx" in refs,
    )


def parse_source(source: str, filename: str = "input.cu") -> list[KernelFacts]:
    """Parse CUDA source text and return facts for every function found."""
    index = ci.Index.create()
    tu = index.parse(
        filename,
        args=_CLANG_ARGS,
        unsaved_files=[(filename, PRELUDE + source)],
        options=ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
    )
    if tu is None:
        raise ParseError(f"clang produced no translation unit for {filename}")

    errors = sum(1 for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error)

    prelude_decls = {
        "__syncthreads",
        "__threadfence",
        "__threadfence_block",
        "fmaxf",
        "fminf",
        "expf",
        "logf",
        "sqrtf",
        "rsqrtf",
        "fabsf",
        *SHUFFLE_NAMES,
        *ATOMIC_NAMES,
    }

    out: list[KernelFacts] = []
    for node in _walk(tu.cursor):
        if node.kind != ci.CursorKind.FUNCTION_DECL:
            continue
        if not node.is_definition() or node.spelling in prelude_decls:
            continue
        facts = _kernel_facts(node)
        facts.parse_errors = errors
        out.append(facts)
    return out


def parse_file(path: str | Path) -> list[KernelFacts]:
    p = Path(path)
    raw = p.read_bytes().decode("utf-8", errors="replace")
    # Strip non-ASCII so stray smart quotes in comments cannot perturb the parse.
    clean = "".join(c if ord(c) < 128 else " " for c in raw)
    return parse_source(clean, filename=p.name)

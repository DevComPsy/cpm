"""Convert the Markdown-flavoured docstrings of cpm to reStructuredText.

The docstrings of cpm were written for mkdocstrings, which renders Markdown.
Sphinx and numpydoc expect reStructuredText, so this script rewrites them:

- docstrings that contain backslashes become raw strings (``\\\\`` -> ``\\``),
- ``$$ ... $$`` blocks become ``.. math::`` directives, ``$...$`` becomes ``:math:``,
- ``[text][cpm.x.Y]`` cross-references become ``:class:``/``:func:`` roles,
  or bare names inside "See Also" sections,
- ``[text](url)`` links become reST links,
- ``_italic_`` becomes ``*italic*``, ``### Heading`` becomes a rubric,
- section names and underlines are normalised for numpydoc,
- blank lines are added around lists.

Usage::

    python docs/scripts/md2rst_docstrings.py --write cpm   # rewrite files
    python docs/scripts/md2rst_docstrings.py --diff cpm    # show the changes
    python docs/scripts/md2rst_docstrings.py --check cpm   # fail on leftover Markdown

The rewrite works on the source text of each docstring literal, never on its
evaluated value, so escape sequences are not silently changed.
"""

import argparse
import ast
import difflib
import importlib
import inspect
import re
import sys
from pathlib import Path

SECTION_RENAMES = {
    "Note": "Notes",
    "See also": "See Also",
    "Reference": "References",
    "Example": "Examples",
}
SECTIONS = {
    "Parameters",
    "Returns",
    "Yields",
    "Receives",
    "Raises",
    "Warns",
    "Other Parameters",
    "Attributes",
    "Methods",
    "See Also",
    "Notes",
    "References",
    "Examples",
    "Warnings",
} | set(SECTION_RENAMES)

CROSSREF = re.compile(r"\[([^\[\]\n]+)\]\[([A-Za-z_][\w.]*)\]")
LINK = re.compile(r"\[([^\[\]\n]+)\]\((https?://[^\s)]+)\)")
INLINE_MATH = re.compile(r"(?<![\\$])\$(?!\$)([^$\n]+?)(?<![\\\s])\$(?!\$)")
ITALIC = re.compile(r"(?<![\w`*/])_([^_\n`]+?)_(?![\w`])")
HEADING = re.compile(r"^(\s*)#{2,6}\s+(.+?)\s*$")
LIST_ITEM = re.compile(r"^(\s*)([-*+]|\d+\.)\s+\S")
INTERSPHINX_PREFIXES = ("scipy.", "numpy.", "pandas.", "matplotlib.")
LF, CRLF = chr(10), chr(13) + chr(10)


# -- Finding docstrings ------------------------------------------------------


def docstring_nodes(tree):
    """Yield the string constants that are docstrings in the module tree."""
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                yield body[0].value


def _offsets(source):
    """Map (line, utf-8 byte column) positions to character offsets."""
    starts, total = [], 0
    lines = source.splitlines(keepends=True)
    for line in lines:
        starts.append(total)
        total += len(line)

    def to_offset(lineno, col):
        line = lines[lineno - 1]
        return starts[lineno - 1] + len(line.encode("utf-8")[:col].decode("utf-8"))

    return to_offset


# -- Literal handling --------------------------------------------------------

LITERAL = re.compile(r'^(?P<prefix>[rRuU]*)(?P<quote>"""|\'\'\'|"|\')(?P<body>.*)(?P=quote)$', re.S)


def split_literal(text):
    match = LITERAL.match(text)
    if not match:
        return None
    return match.group("prefix"), match.group("quote"), match.group("body")


def make_raw(prefix, body):
    """Turn a non-raw docstring body with backslashes into a raw one."""
    if "r" in prefix.lower() or "\\" not in body:
        return prefix, body
    body = body.replace("\\\\", "\x00").replace("\x00", "\\")
    return "r" + prefix, body


# -- Body transformations ----------------------------------------------------


def resolve_role(target):
    """Pick a Sphinx role for a dotted Python name by importing it."""
    parts = target.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError:
            return "obj"
        if inspect.ismodule(obj):
            return "mod"
        if inspect.isclass(obj):
            return "class"
        if inspect.isfunction(obj) and i < len(parts) - 1:
            return "meth"
        if callable(obj):
            return "func"
        return "obj"
    return "obj"


def convert_crossref(match):
    text, target = match.group(1).strip("`"), match.group(2)
    role = resolve_role(target)
    if text == target:
        return f":{role}:`{target}`"
    return f":{role}:`{text} <{target}>`"


def convert_link(match):
    text, url = match.group(1), match.group(2)
    name = text.strip("`")
    if text.startswith("`") and name.startswith(INTERSPHINX_PREFIXES):
        return f":func:`{name}`" if resolve_role(name) != "class" else f":class:`{name}`"
    return f"`{name} <{url}>`__"


def convert_inline(line):
    line = CROSSREF.sub(convert_crossref, line)
    line = LINK.sub(convert_link, line)
    line = INLINE_MATH.sub(lambda m: f":math:`{m.group(1).strip()}`", line)
    if ":math:" not in line and "``" not in line:
        line = ITALIC.sub(r"*\1*", line)
    return line


def is_underline(line):
    stripped = line.strip()
    return len(stripped) >= 3 and set(stripped) == {"-"}


def normalise_sections(lines):
    out = []
    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        title = line.strip()
        if title in SECTIONS and is_underline(nxt):
            title = SECTION_RENAMES.get(title, title)
            indent = line[: len(line) - len(line.lstrip())]
            out.append(indent + title)
            lines[i + 1] = indent + "-" * len(title)
            continue
        out.append(line)
    return out


def convert_display_math(lines):
    """Turn $$ ... $$ blocks into .. math:: directives."""
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        indent = line[: len(line) - len(line.lstrip())]
        if stripped.startswith("$$"):
            if stripped != "$$" and stripped.endswith("$$") and len(stripped) > 4:
                content = [stripped[2:-2].strip()]
                i += 1
            else:
                content = []
                first = stripped[2:].strip()
                if first:
                    content.append(first)
                i += 1
                while i < len(lines) and not lines[i].strip().endswith("$$"):
                    content.append(lines[i].strip())
                    i += 1
                if i < len(lines):
                    last = lines[i].strip()[:-2].strip()
                    if last:
                        content.append(last)
                    i += 1
            if out and out[-1].strip():
                out.append("")
            out.append(f"{indent}.. math::")
            out.append("")
            out.extend(f"{indent}    {c}" if c else "" for c in content)
            out.append("")
            if i < len(lines) and not lines[i].strip():
                i += 1
            continue
        out.append(line)
        i += 1
    return out


def see_also_entry(line):
    """Rewrite '[x][x] : description' as numpydoc's 'x : description'."""
    match = re.match(r"^(\s*)\[([^\[\]]+)\]\[([\w.]+)\]\s*:?\s*(.*)$", line)
    if not match:
        return None
    indent, _, target, desc = match.groups()
    return f"{indent}{target} : {desc}" if desc else f"{indent}{target}"


def convert_body(body):
    lines = body.split("\n")
    lines = normalise_sections(lines)
    lines = convert_display_math(lines)
    out, section, in_math = [], None, False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(".. math::"):
            in_math = True
            math_indent = len(line) - len(line.lstrip())
            out.append(line)
            continue
        if in_math:
            if stripped and len(line) - len(line.lstrip()) <= math_indent:
                in_math = False
            else:
                out.append(line)
                continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if is_underline(nxt) and stripped in SECTIONS:
            section = stripped
        heading = HEADING.match(line)
        if heading:
            if out and out[-1].strip():
                out.append("")
            out.append(f"{heading.group(1)}.. rubric:: {heading.group(2)}")
            if nxt.strip():
                out.append("")
            continue
        if section == "See Also":
            entry = see_also_entry(line)
            if entry is not None:
                out.append(entry)
                continue
        out.append(convert_inline(line))
    return "\n".join(separate_lists(out))


def separate_lists(lines):
    """Add the blank lines that reST needs around bullet and numbered lists."""
    out = []
    for i, line in enumerate(lines):
        item = LIST_ITEM.match(line)
        prev = out[-1] if out else ""
        if item and prev.strip() and not LIST_ITEM.match(prev) and not _is_continuation(prev, out):
            out.append("")
        elif (
            not item
            and line.strip()
            and prev.strip()
            and (LIST_ITEM.match(prev) or _is_continuation(prev, out))
            and _indent(line) <= _list_indent(out)
        ):
            out.append("")
        out.append(line)
    return out


def _indent(line):
    return len(line) - len(line.lstrip())


def _list_indent(out):
    for line in reversed(out):
        if not line.strip():
            return -1
        match = LIST_ITEM.match(line)
        if match:
            return len(match.group(1))
    return -1


def _is_continuation(prev, out):
    """Whether prev is a wrapped line that belongs to a list item."""
    indent = _list_indent(out)
    return indent >= 0 and _indent(prev) > indent


# -- Driver ------------------------------------------------------------------

LEFTOVERS = [
    (re.compile(r"(?<!\\)\$"), "dollar math"),
    (re.compile(r"\]\[[\w.]+\]"), "Markdown cross-reference"),
    (re.compile(r"\]\(https?://"), "Markdown link"),
]


def convert_source(source):
    tree = ast.parse(source)
    to_offset = _offsets(source)
    edits = []
    for node in docstring_nodes(tree):
        start = to_offset(node.lineno, node.col_offset)
        end = to_offset(node.end_lineno, node.end_col_offset)
        parts = split_literal(source[start:end])
        if parts is None:
            print(f"  skipped a docstring at line {node.lineno}", file=sys.stderr)
            continue
        prefix, quote, body = parts
        prefix, body = make_raw(prefix, body)
        new = prefix + quote + convert_body(body) + quote
        if new != source[start:end]:
            edits.append((start, end, new))
    for start, end, new in sorted(edits, reverse=True):
        source = source[:start] + new + source[end:]
    return source


def check_source(path, source):
    problems = []
    for node in docstring_nodes(ast.parse(source)):
        to_offset = _offsets(source)
        text = source[to_offset(node.lineno, node.col_offset) : to_offset(node.end_lineno, node.end_col_offset)]
        parts = split_literal(text)
        if parts is None:
            continue
        prefix, _, body = parts
        if "\\" in body and "r" not in prefix.lower():
            problems.append((node.lineno, "backslash in a non-raw docstring"))
        for pattern, name in LEFTOVERS:
            for match in pattern.finditer(body):
                line = node.lineno + body[: match.start()].count("\n")
                problems.append((line, name))
    return [f"{path}:{line}: {name}" for line, name in problems]


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--diff", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)

    files = sorted(
        f for p in args.paths for f in ([p] if p.is_file() else p.rglob("*.py"))
    )
    failures = []
    for path in files:
        raw = path.read_bytes()
        newline = CRLF if CRLF.encode() in raw else LF
        source = raw.decode("utf-8").replace(CRLF, LF)
        if args.check:
            failures += check_source(path, source)
            continue
        new = convert_source(source)
        if new == source:
            continue
        if args.diff:
            sys.stdout.writelines(
                difflib.unified_diff(
                    source.splitlines(keepends=True),
                    new.splitlines(keepends=True),
                    str(path),
                    str(path),
                )
            )
        else:
            path.write_bytes(new.replace(LF, newline).encode("utf-8"))
            print(f"rewrote {path}")
    if failures:
        print("\n".join(failures))
        print(f"{len(failures)} Markdown leftovers in docstrings", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

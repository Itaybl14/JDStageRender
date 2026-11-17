"""
Minimal parser for the UbiArt "Lua" data files (.tape / .tpl / .trk / .isc-lua
exports such as the ones produced by ubiart2lua or the editor).

These files are *data*, not programs: a sequence of `name = <expr>` statements
where <expr> is a table constructor, number, string, boolean, a reference to
an earlier variable (e.g. `structure = structure` after an `.trk` include) or a
call we can ignore (`includeReference("...")`).

`normalize()` converts the Lua shape to the same shape the cooked JSON (.ckd)
files use, so the rest of the renderer only deals with one representation:

  { NAME = "Clip", Clip = { ... } }   ->  {"__class": "Clip", ...}
  { MusicSignature = { ... } }        ->  {"__class": "MusicSignature", ...}
  { VAL = 3 }                         ->  3
  array-style tables                  ->  lists
"""

import re

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+|--\[\[.*?\]\]|--[^\n]*) |
    (?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*') |
    (?P<num>-?(?:0[xX][0-9a-fA-F]+|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)) |
    (?P<name>[A-Za-z_][A-Za-z0-9_]*) |
    (?P<op>[{}\[\]=,;()])
    """,
    re.VERBOSE | re.DOTALL,
)


class LuaParseError(ValueError):
    pass


def _tokenize(text):
    pos = 0
    out = []
    n = len(text)
    while pos < n:
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise LuaParseError(f"unexpected character {text[pos]!r} at offset {pos}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "ws":
            continue
        out.append((kind, m.group(kind)))
    out.append(("eof", None))
    return out


class _Parser:
    def __init__(self, text):
        self.toks = _tokenize(text)
        self.i = 0
        self.env = {}

    def peek(self, k=0):
        return self.toks[self.i + k]

    def next(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, val):
        t = self.next()
        if t[1] != val:
            raise LuaParseError(f"expected {val!r}, got {t[1]!r}")
        return t

    def parse_chunk(self):
        while self.peek()[0] != "eof":
            kind, val = self.peek()
            if kind == "op" and val == ";":
                self.next()
                continue
            if kind == "name" and self.peek(1)[1] == "=":
                self.next()
                self.next()
                self.env[val] = self.parse_expr()
            elif kind == "name" and self.peek(1)[1] == "(":
                self.parse_expr()  # function call statement, e.g. includeReference(...)
            else:
                raise LuaParseError(f"unexpected token {val!r}")
        return self.env

    def parse_expr(self):
        kind, val = self.next()
        if kind == "op" and val == "{":
            return self.parse_table()
        if kind == "str":
            body = val[1:-1]
            return bytes(body, "latin1").decode("unicode_escape") if "\\" in body else body
        if kind == "num":
            if val.lower().startswith(("0x", "-0x")):
                return int(val, 16)
            f = float(val)
            return int(f) if re.fullmatch(r"-?\d+", val) else f
        if kind == "name":
            if val == "true":
                return True
            if val == "false":
                return False
            if val == "nil":
                return None
            if self.peek()[1] == "(":  # call: skip arguments, return None
                self.next()
                depth = 1
                while depth:
                    t = self.next()
                    if t[0] == "eof":
                        raise LuaParseError("unterminated call")
                    if t[1] == "(":
                        depth += 1
                    elif t[1] == ")":
                        depth -= 1
                return None
            return self.env.get(val)
        raise LuaParseError(f"unexpected token {val!r} in expression")

    def parse_table(self):
        keyed = {}
        array = []
        order = []
        while True:
            kind, val = self.peek()
            if kind == "op" and val == "}":
                self.next()
                break
            if kind == "op" and val in ",;":
                self.next()
                continue
            if kind == "name" and self.peek(1)[1] == "=":
                self.next()
                self.next()
                keyed[val] = self.parse_expr()
                order.append(val)
            elif kind == "op" and val == "[":
                self.next()
                key = self.parse_expr()
                self.expect("]")
                self.expect("=")
                keyed[key] = self.parse_expr()
                order.append(key)
            else:
                array.append(self.parse_expr())
        if array and not keyed:
            return array
        if keyed and not array:
            return keyed
        if not keyed and not array:
            return {}
        keyed["__array__"] = array
        return keyed


def parse(text):
    """Parse a UbiArt Lua data file and return its global variables."""
    return _Parser(text).parse_chunk()


def normalize(obj):
    """Convert a parsed Lua value to the cooked-JSON shape (see module doc)."""
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    if not isinstance(obj, dict):
        return obj
    if set(obj.keys()) == {"VAL"}:
        return normalize(obj["VAL"])
    name = obj.get("NAME")
    if isinstance(name, str) and name in obj and isinstance(obj[name], dict) and len(obj) == 2:
        inner = normalize(obj[name])
        if isinstance(inner, dict):
            return {"__class": name, **inner}
        return inner
    if len(obj) == 1:
        (k, v), = obj.items()
        if isinstance(k, str) and k[:1].isupper() and isinstance(v, dict):
            inner = normalize(v)
            if isinstance(inner, dict):
                return {"__class": k, **inner}
    return {k: normalize(v) for k, v in obj.items() if k != "__array__"}


def load_lua_data(text):
    """Return the normalized `params` table (or the whole env if absent)."""
    env = parse(text)
    root = env.get("params", env)
    return normalize(root)

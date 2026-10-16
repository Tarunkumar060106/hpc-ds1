"""Terminal styling for the CLI: colour, symbols and a few layout helpers.

No dependencies. Colour is used only when writing to a terminal, and is
turned off by NO_COLOR (https://no-color.org) or forced on by FORCE_COLOR.
On Windows the console's ANSI mode is switched on when needed.
"""

import os
import re
import sys

_CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "blue": "34",
          "magenta": "35", "cyan": "36", "grey": "90"}


def _enable_windows_ansi():
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)  # stdout
        mode = ctypes.c_uint32()
        if k.GetConsoleMode(h, ctypes.byref(mode)):
            k.SetConsoleMode(h, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        return True
    except Exception:
        return False


def _use_colour():
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()):
        return False
    return _enable_windows_ansi() if os.name == "nt" else True


def _can_print(text):
    try:
        text.encode(sys.stdout.encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


COLOUR = _use_colour()
UNICODE = _can_print("✓✗●→─")
OK, BAD, WARN, DOT, ARROW, RULE = ("✓", "✗", "!", "●", "→", "─") if UNICODE else ("+", "x", "!", "*", "->", "-")


def paint(text, *styles):
    if not COLOUR or not styles:
        return str(text)
    return "\033[" + ";".join(_CODES[s] for s in styles) + "m" + str(text) + "\033[0m"


def bold(t): return paint(t, "bold")
def dim(t): return paint(t, "grey")
def good(t): return paint(t, "green")
def warn(t): return paint(t, "yellow")
def bad(t): return paint(t, "red")
def accent(t): return paint(t, "cyan")
def num(t): return paint(t, "bold", "cyan")


MODE_STYLE = {"auto": ("green", "AUTO"), "confirm": ("yellow", "CONFIRM"), "escalate": ("red", "ESCALATE")}


def mode(m):
    """AUTO / CONFIRM / ESCALATE badge."""
    colour, label = MODE_STYLE.get(str(m).lower(), ("grey", str(m).upper()))
    if not COLOUR:
        return f"[{label}]"
    bg = {"green": "42", "yellow": "43", "red": "41", "grey": "100"}[colour]
    return f"[1;30;{bg}m {label} [0m"


def cfg(c):
    """{'ranks': 1, 'threads': 8, 'gpu': True, 'precision': 'bf16'} -> 'GPU · 8 threads · bf16'."""
    if not c:
        return ""
    sep = " · " if UNICODE else " | "
    if c.get("gpu"):
        parts = [paint("GPU", "bold", "magenta"), f"{c['threads']} thread{'s' if c['threads'] != 1 else ''}"]
    else:
        parts = [paint("CPU", "bold", "blue"), f"{c['ranks']} rank{'s' if c['ranks'] != 1 else ''} × {c['threads']} thread{'s' if c['threads'] != 1 else ''}"
                 if UNICODE else f"{c['ranks']} ranks x {c['threads']} threads"]
    return sep.join(parts + [c["precision"]])


def secs(s):
    """0.00235 -> '2.35 ms'; 4.2 -> '4.20 s'."""
    if s is None:
        return "-"
    return f"{s * 1e3:.2f} ms" if s < 1 else f"{s:.2f} s"


def pct(x):
    return f"{x:.0%}"


def header(title, sub=None):
    line = paint(RULE * 2, "grey")
    print(f"\n{line} {bold(title)}" + (f"  {dim(sub)}" if sub else ""), flush=True)


def kv(key, value, width=18):
    print(f"  {dim(key.ljust(width))} {value}", flush=True)


def item(text, kind="dot"):
    sym = {"ok": good(OK), "bad": bad(BAD), "warn": warn(WARN), "dot": accent(DOT), "arrow": accent(ARROW)}[kind]
    print(f"  {sym} {text}", flush=True)


# ---------------------------------------------------------------------------
# progress messages from setup / learn / the fleet agent
# ---------------------------------------------------------------------------

_STEP = re.compile(r"^(\s*)\[(\d+/\d+|\+)\]\s*(.*)$")
_PCT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?%)")
_CFG = re.compile(r"\{'ranks': (\d+), 'threads': (\d+), 'gpu': (True|False), 'precision': '(\w+)'\}")


def _cfgs(text):
    return _CFG.sub(lambda m: cfg({"ranks": int(m[1]), "threads": int(m[2]), "gpu": m[3] == "True",
                                   "precision": m[4]}), text)


def log(message):
    """Print one progress line with colour picked from its content."""
    m = str(message)
    step = _STEP.match(m)
    low = m.lower()
    if step:
        indent, n, rest = step.groups()
        out = f"{indent}{paint('[' + n + ']', 'bold', 'cyan')} {bold(rest)}"
    elif "failed" in low or "error" in low or "can't" in low or "rejected" in low:
        out = f"{bad(BAD)} {_cfgs(m.strip())}" if not m.startswith(" ") else m[:len(m) - len(m.lstrip())] + bad(m.strip())
    elif low.startswith(("note:", "warning")) or "not this machine" in low:
        out = f"{warn(WARN)} {warn(m)}"
    elif low.startswith(("saved", "joined")):
        out = f"{good(OK)} {_cfgs(m)}"
    elif low.startswith("task "):
        tid, _, rest = m.partition(":")
        rest = _cfgs(rest)
        rest = re.sub(r"\((\d+%) sure, (auto|confirm|escalate)\)", lambda x: f"{num(x[1])} sure {mode(x[2])}", rest)
        rest = re.sub(r"(x[\d.]+ vs untuned)", lambda x: good(bold(x[1])), rest)
        out = f"{accent(DOT)} {bold(tid)}{dim(':')}{rest}"
    else:
        out = _cfgs(m)
    out = _PCT.sub(lambda x: num(x[1]), out) if "\033[" not in out[:12] or step else out
    print(out, flush=True)

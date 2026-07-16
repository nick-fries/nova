# Copyright 2026 OpenStack Foundation
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

"""Operator-controlled libvirt domain XML overrides (VDI latency tuning).

Applied as the final step of ``LibvirtDriver._get_guest_xml()``.  Driven by
flavor extra specs in the ``vdi:`` namespace; large overrides live in
Jinja-templated YAML profile files under ``[vdi]profile_dir``.  See
nova-domain-xml-override-design.md for the full specification.

Flavor keys:

* ``vdi:profile=<name>[,<name>...]`` -- profile files, applied in order.
* ``vdi:xml.<NN>=<xpath>|<op-spec>|<payload>`` -- inline micro-ops, applied
  in numeric NN order after all profiles.  ``<op-spec>`` is
  ``<op>[@<match>][:<attr-name>]``.  Literal pipes are escaped ``\\|``.
* ``vdi:flag.<word>=<value>`` -- free-form parameters, ignored by the
  engine, visible to profile templates via the ``flag()`` helper.

Fail-closed: any malformed key, missing profile, template error, xpath
cardinality violation, denied xpath or unparsable fragment raises
``VDIXMLOverrideError`` and aborts the render.  The single soft failure is
the ``[vdi]enabled = False`` kill switch, which logs a WARNING and returns
the stock XML.
"""

import argparse
import copy
import dataclasses
import difflib
import os
import re
import sys

import jinja2
import jinja2.sandbox
from lxml import etree
from oslo_config import cfg
from oslo_log import log as logging
import yaml

from nova import exception

LOG = logging.getLogger(__name__)

QEMU_NS = "http://libvirt.org/schemas/domain/qemu/1.0"
NSMAP = {"qemu": QEMU_NS}

FRAGMENT_OPS = frozenset(
    {"replace", "upsert", "append-xml", "prepend-xml",
     "insert-before", "insert-after"})
ATTR_OPS = frozenset({"set-attr", "remove-attr"})
ALL_OPS = FRAGMENT_OPS | ATTR_OPS | {"set-text", "remove"}
MATCH_MODES = frozenset({"one", "all", "opt", "any"})

# Scheduler-visible resources, identity, and live-migration-rewriter-owned
# regions.  Extending this list is config (`[vdi]extra_denied_xpaths`);
# shrinking it is a deliberate one-line fork decision, not a config knob.
DENIED_XPATH_PREFIXES = (
    "/domain/name", "/domain/uuid",
    "/domain/memory", "/domain/currentMemory", "/domain/vcpu",
    "/domain/cputune/vcpupin", "/domain/cputune/emulatorpin",
    "/domain/cputune/iothreadpin",
    "/domain/numatune", "/domain/cpu/numa",
    "/domain/memoryBacking/hugepages",
    "/domain/devices/disk", "/domain/devices/interface",
)

PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MICRO_OP_KEY_RE = re.compile(r"^vdi:xml\.(\d+)$")
FLAG_KEY_RE = re.compile(r"^vdi:flag\.[A-Za-z0-9_.-]+$")
OP_SPEC_RE = re.compile(
    r"^(?P<op>[a-z-]+)(@(?P<match>[a-z-]+))?(:(?P<arg>.+))?$")
_PREDICATE_RE = re.compile(r"\[[^\]]*\]")

_PROFILE_SCHEMA_TOP = frozenset({"description", "namespaces", "ops"})
_PROFILE_SCHEMA_OP = frozenset(
    {"xpath", "op", "match", "name", "value", "xml"})

# Registered in-module rather than in nova/conf/ to keep the fork's
# touched-file count minimal (design doc section 8.3).
vdi_opts = [  # noqa: N342
    cfg.BoolOpt("enabled", default=True,
                help="Kill switch. False: vdi:* keys are ignored with a "
                     "WARNING and instances boot on stock XML."),
    cfg.StrOpt("profile_dir", default="/etc/nova/vdi-profiles",
               help="Directory containing <name>.yaml.j2 profile files."),
    cfg.IntOpt("max_ops", default=64, min=1,
               help="Maximum patch ops per instance render."),
    cfg.IntOpt("max_fragment_bytes", default=16384, min=1,
               help="Maximum size of a single XML fragment."),
    cfg.ListOpt("extra_denied_xpaths", default=[],
                help="Additional denied xpath prefixes (extends, never "
                     "shrinks, the hardcoded deny-list)."),
    cfg.BoolOpt("log_diff", default=True,
                help="Log a base->final unified diff at INFO per render."),
]
CONF = cfg.CONF
CONF.register_opts(vdi_opts, group="vdi")


class VDIXMLOverrideError(exception.NovaException):
    msg_fmt = "VDI XML override failed: %(reason)s"


def _err(reason, source=None):
    if source:
        reason = "%s: %s" % (source, reason)
    return VDIXMLOverrideError(reason=reason)


@dataclasses.dataclass(frozen=True)
class PatchOp:
    """One patch operation (see design doc section 4.1)."""

    xpath: str          # target selector, evaluated against current tree
    op: str             # one of ALL_OPS
    match: str = "one"  # cardinality contract: one | all | opt | any
    name: str = None    # attribute name  (set-attr, remove-attr)
    value: str = None   # attr/text value (set-attr, set-text)
    xml: str = None     # XML fragment    (fragment ops)
    source: str = ""    # provenance, e.g. "profile:latency-gold[3]"


def apply(xml, instance, image_meta):
    """Apply operator XML overrides.  str -> str.

    Entry point called from ``LibvirtDriver._get_guest_xml``.  Returns the
    input unchanged (same object, no parse) when the instance's flavor
    carries no ``vdi:`` extra specs.
    """
    # Tolerate legacy callers/tests that pass dict-like instances or
    # flavors without extra_specs loaded: no specs means fast path.
    flavor = getattr(instance, "flavor", None)
    specs = getattr(flavor, "extra_specs", None) or {}
    if not any(k.startswith("vdi:") for k in specs):
        return xml
    if not CONF.vdi.enabled:
        LOG.warning("[vdi]enabled=False: ignoring vdi:* extra specs; "
                    "instance boots on stock XML.", instance=instance)
        return xml

    vdi_keys = _collect_vdi_keys(specs)

    def context_builder(base_tree):
        return _build_jinja_context(instance, image_meta, base_tree)

    final = _render(
        xml, vdi_keys, context_builder,
        profile_dir=CONF.vdi.profile_dir,
        max_ops=CONF.vdi.max_ops,
        max_fragment_bytes=CONF.vdi.max_fragment_bytes,
        extra_denied=tuple(CONF.vdi.extra_denied_xpaths))
    if CONF.vdi.log_diff:
        _log_diff(instance, xml, final)
    return final


def _render(base_xml, vdi_keys, context_builder, profile_dir, max_ops,
            max_fragment_bytes, extra_denied=()):
    """Shared core used by apply() and the dry-run CLI."""
    try:
        tree = etree.fromstring(base_xml.encode("utf-8"))
    except etree.XMLSyntaxError as e:
        raise _err("base XML failed to parse: %s" % e)
    base_tree = copy.deepcopy(tree)  # pristine copy for the base() helper

    context = context_builder(base_tree)

    nsmap = dict(NSMAP)
    ops = []
    profile_names = [n.strip()
                     for n in vdi_keys.get("vdi:profile", "").split(",")
                     if n.strip()]
    for name in profile_names:
        p_ops, p_ns = _load_profile(name, context, profile_dir,
                                    max_fragment_bytes)
        for prefix, uri in p_ns.items():
            if nsmap.get(prefix, uri) != uri:
                raise _err("namespace prefix '%s' redefined with a "
                           "different URI" % prefix,
                           source="profile:%s" % name)
            nsmap[prefix] = uri
        ops.extend(p_ops)
    ops.extend(_parse_micro_ops(vdi_keys, max_fragment_bytes))

    if not ops:
        raise _err("vdi:* keys present but no ops resolved (empty "
                   "vdi:profile and no vdi:xml.NN keys)")
    if len(ops) > max_ops:
        raise _err("op count %d exceeds [vdi]max_ops=%d"
                   % (len(ops), max_ops))

    denied = DENIED_XPATH_PREFIXES + tuple(extra_denied)
    for op in ops:
        _deny_check(op.xpath, denied, op.source)
        if op.op == "upsert":
            _deny_check(_upsert_parent_xpath(op), denied, op.source)
        _apply_op(tree, op, nsmap)

    final = _serialize(tree, nsmap)
    LOG.info("vdi xml_override applied %(nops)d op(s) "
             "(profiles: %(profiles)s)",
             {"nops": len(ops),
              "profiles": ",".join(profile_names) or "-"})
    return final


# -- collation ---------------------------------------------------------

def _collect_vdi_keys(extra_specs):
    """Filter and validate vdi:* keys (error #4 for unknown keys)."""
    vdi_keys = {}
    for key, value in extra_specs.items():
        if not key.startswith("vdi:"):
            continue
        if (key != "vdi:profile" and
                not MICRO_OP_KEY_RE.match(key) and
                not FLAG_KEY_RE.match(key)):
            raise _err("unknown extra spec key '%s' (expected vdi:profile, "
                       "vdi:xml.NN or vdi:flag.<word>)" % key)
        vdi_keys[key] = value
    return vdi_keys


def _split_unescaped_pipes(value):
    """Split on unescaped '|'; unescape '\\|' in the parts."""
    parts = []
    cur = []
    i = 0
    while i < len(value):
        ch = value[i]
        if ch == "\\" and i + 1 < len(value) and value[i + 1] == "|":
            cur.append("|")
            i += 2
        elif ch == "|":
            parts.append("".join(cur))
            cur = []
            i += 1
        else:
            cur.append(ch)
            i += 1
    parts.append("".join(cur))
    return parts


def _parse_micro_ops(vdi_keys, max_fragment_bytes):
    """Parse vdi:xml.NN keys into PatchOps, numeric order (errors #5)."""
    numbered = {}
    for key, value in vdi_keys.items():
        m = MICRO_OP_KEY_RE.match(key)
        if not m:
            continue
        source = "flavor:%s" % key
        nn = int(m.group(1))
        if not 1 <= nn <= 99:
            raise _err("micro-op number must be 1-99", source=source)
        if nn in numbered:
            raise _err("duplicate micro-op number %d (e.g. vdi:xml.2 vs "
                       "vdi:xml.02)" % nn, source=source)
        numbered[nn] = (source, value)

    ops = []
    for nn in sorted(numbered):
        source, value = numbered[nn]
        parts = _split_unescaped_pipes(value)
        if len(parts) != 3:
            raise _err("expected <xpath>|<op-spec>|<payload> (exactly two "
                       "unescaped pipes; escape literals as \\|), got %d "
                       "field(s)" % len(parts), source=source)
        xpath, op_spec, payload = parts
        if not xpath:
            raise _err("empty xpath", source=source)
        m = OP_SPEC_RE.match(op_spec)
        if not m:
            raise _err("malformed op-spec '%s' (expected "
                       "<op>[@<match>][:<arg>])" % op_spec, source=source)
        op = m.group("op")
        match = m.group("match") or "one"
        arg = m.group("arg")
        if op not in ALL_OPS:
            raise _err("unknown op '%s' (known: %s)"
                       % (op, ", ".join(sorted(ALL_OPS))), source=source)
        if match not in MATCH_MODES:
            raise _err("unknown match mode '%s' (known: %s)"
                       % (match, ", ".join(sorted(MATCH_MODES))),
                       source=source)
        kwargs = {"xpath": xpath, "op": op, "match": match,
                  "source": source}
        if op in ATTR_OPS:
            if not arg:
                raise _err("op '%s' requires an attribute name "
                           "(':<name>')" % op, source=source)
            kwargs["name"] = arg
        elif arg:
            raise _err("op '%s' does not take an ':<arg>'" % op,
                       source=source)
        if op in FRAGMENT_OPS:
            kwargs["xml"] = payload
        elif op in ("set-attr", "set-text"):
            kwargs["value"] = payload
        elif payload:
            raise _err("op '%s' takes an empty payload" % op,
                       source=source)
        pop = PatchOp(**kwargs)
        _validate_op(pop, max_fragment_bytes)
        ops.append(pop)
    return ops


def _validate_op(op, max_fragment_bytes):
    """Shared structural validation for micro-ops and profile ops."""
    if op.op not in ALL_OPS:
        raise _err("unknown op '%s' (known: %s)"
                   % (op.op, ", ".join(sorted(ALL_OPS))), source=op.source)
    if op.match not in MATCH_MODES:
        raise _err("unknown match mode '%s' (known: %s)"
                   % (op.match, ", ".join(sorted(MATCH_MODES))),
                   source=op.source)
    if op.op in FRAGMENT_OPS:
        if not op.xml or not op.xml.strip():
            raise _err("op '%s' requires an XML fragment" % op.op,
                       source=op.source)
        if len(op.xml.encode("utf-8")) > max_fragment_bytes:
            raise _err("fragment exceeds [vdi]max_fragment_bytes=%d"
                       % max_fragment_bytes, source=op.source)
    if op.op in ATTR_OPS and not op.name:
        raise _err("op '%s' requires an attribute name" % op.op,
                   source=op.source)
    if op.op in ("set-attr", "set-text") and op.value is None:
        raise _err("op '%s' requires a value" % op.op, source=op.source)


def _load_profile(name, context, profile_dir, max_fragment_bytes):
    """Load one profile file -> ([PatchOp], namespaces) (errors #1-3)."""
    source = "profile:%s" % name
    if not PROFILE_NAME_RE.match(name):
        raise _err("invalid profile name (must match %s)"
                   % PROFILE_NAME_RE.pattern, source=source)
    path = None
    for suffix in (".yaml.j2", ".yaml"):
        candidate = os.path.join(profile_dir, name + suffix)
        if (os.path.realpath(candidate).startswith(
                os.path.realpath(profile_dir) + os.sep) and
                os.path.isfile(candidate)):
            path = candidate
            break
    if path is None:
        raise _err("profile '%s' not found in %s" % (name, profile_dir),
                   source=source)

    env = jinja2.sandbox.SandboxedEnvironment(
        loader=jinja2.FileSystemLoader(profile_dir),
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True)
    try:
        template = env.get_template(os.path.basename(path))
        rendered = template.render(**context)
    except jinja2.exceptions.TemplateError as e:
        raise _err("template error: %s" % e, source=source)

    try:
        data = yaml.safe_load(rendered)
    except yaml.YAMLError as e:
        raise _err("rendered profile is not valid YAML: %s" % e,
                   source=source)

    if not isinstance(data, dict):
        raise _err("profile must be a YAML mapping", source=source)
    unknown = set(data) - _PROFILE_SCHEMA_TOP
    if unknown:
        raise _err("unknown top-level key(s): %s"
                   % ", ".join(sorted(unknown)), source=source)
    namespaces = data.get("namespaces") or {}
    if (not isinstance(namespaces, dict) or
            not all(isinstance(k, str) and isinstance(v, str)
                    for k, v in namespaces.items())):
        raise _err("'namespaces' must map prefix strings to URI strings",
                   source=source)
    raw_ops = data.get("ops")
    if not isinstance(raw_ops, list) or not raw_ops:
        raise _err("'ops' must be a non-empty list", source=source)

    ops = []
    for i, raw in enumerate(raw_ops):
        op_source = "profile:%s[%d]" % (name, i)
        if not isinstance(raw, dict):
            raise _err("op must be a mapping", source=op_source)
        unknown = set(raw) - _PROFILE_SCHEMA_OP
        if unknown:
            raise _err("unknown op key(s): %s"
                       % ", ".join(sorted(unknown)), source=op_source)
        if not raw.get("xpath") or not raw.get("op"):
            raise _err("'xpath' and 'op' are required", source=op_source)
        pop = PatchOp(
            xpath=str(raw["xpath"]).strip(),
            op=str(raw["op"]),
            match=str(raw.get("match", "one")),
            name=raw.get("name"),
            value=(None if raw.get("value") is None
                   else str(raw["value"])),
            xml=raw.get("xml"),
            source=op_source)
        _validate_op(pop, max_fragment_bytes)
        ops.append(pop)
    if data.get("description"):
        LOG.debug("loaded vdi profile %s: %s", name,
                  str(data["description"]).strip())
    return ops, namespaces


def _iget(obj, attr, default=None):
    """Read a versioned-object attribute, tolerating unset fields."""
    try:
        is_set = obj.obj_attr_is_set(attr)
    except (AttributeError, KeyError):
        is_set = True
    if not is_set:
        return default
    return getattr(obj, attr, default)


def _build_jinja_context(instance, image_meta, base_tree):
    """The complete, deliberately small template context (design 6.4)."""
    flavor = instance.flavor
    specs = dict(flavor.extra_specs or {})

    image = {}
    image_name = None
    image_id = None
    if image_meta is not None:
        props = _iget(image_meta, "properties")
        if props is not None:
            for field in props.obj_fields:
                if props.obj_attr_is_set(field):
                    image[field] = getattr(props, field)
        image_name = _iget(image_meta, "name")
        image_id = _iget(image_meta, "id")
    image["image_name"] = image_name
    image["image_id"] = image_id

    def flag(name, default=""):
        return specs.get("vdi:flag." + name, default)

    def base(xpath):
        try:
            results = etree.XPath(xpath, namespaces=NSMAP)(base_tree)
        except (etree.XPathSyntaxError, etree.XPathEvalError) as e:
            raise _err("base() helper: bad xpath '%s': %s" % (xpath, e))
        if not isinstance(results, list):
            return [str(results)]
        out = []
        for r in results:
            if isinstance(r, etree._Element):
                out.append(r.text or "")
            else:
                out.append(str(r))
        return out

    return {
        "instance": {
            "uuid": _iget(instance, "uuid"),
            "name": _iget(instance, "display_name"),
            "hostname": _iget(instance, "hostname"),
            "os_type": _iget(instance, "os_type"),
            "vcpus": _iget(instance, "vcpus"),
            "memory_mb": _iget(instance, "memory_mb"),
        },
        "flavor": {
            "name": _iget(flavor, "name"),
            "flavorid": _iget(flavor, "flavorid"),
            "vcpus": _iget(flavor, "vcpus"),
            "memory_mb": _iget(flavor, "memory_mb"),
            "extra_specs": specs,
        },
        "specs": specs,
        "image": image,
        "flag": flag,
        "base": base,
        "host": CONF.host,
    }


# -- engine ------------------------------------------------------------

def _canonical_path(xpath):
    """Strip [predicates] for deny-list prefix comparison."""
    prev = None
    canon = xpath
    while canon != prev:
        prev = canon
        canon = _PREDICATE_RE.sub("", canon)
    return canon.rstrip("/") or "/"


def _deny_check(xpath, denied, source):
    canon = _canonical_path(xpath)
    for prefix in denied:
        if canon == prefix or canon.startswith(prefix + "/"):
            raise _err("xpath '%s' targets denied region '%s'"
                       % (xpath, prefix), source=source)


def _upsert_parent_xpath(op):
    """xpath minus its final location step (bracket-aware)."""
    depth = 0
    idx = -1
    for i, ch in enumerate(op.xpath):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "/" and depth == 0:
            idx = i
    if idx <= 0:
        raise _err("cannot upsert the document root (xpath '%s' has no "
                   "parent step)" % op.xpath, source=op.source)
    return op.xpath[:idx]


def _xpath_elements(tree, xpath, nsmap, source):
    """Evaluate an xpath; require an element-only node-set (error #8)."""
    try:
        results = etree.XPath(xpath, namespaces=nsmap)(tree)
    except (etree.XPathSyntaxError, etree.XPathEvalError) as e:
        raise _err("invalid xpath '%s': %s" % (xpath, e), source=source)
    if not isinstance(results, list) or not all(
            isinstance(r, etree._Element) for r in results):
        raise _err("xpath '%s' must select elements" % xpath,
                   source=source)
    return results


def _check_cardinality(op, nodes):
    n = len(nodes)
    detail = "xpath '%s' matched %d element(s)" % (op.xpath, n)
    if op.match == "one" and n != 1:
        raise _err("%s, match=one requires exactly 1" % detail,
                   source=op.source)
    if op.match == "all" and n < 1:
        raise _err("%s, match=all requires at least 1" % detail,
                   source=op.source)
    if op.match == "opt" and n > 1:
        raise _err("%s, match=opt allows at most 1" % detail,
                   source=op.source)


def _parse_fragment(op, nsmap):
    """Parse op.xml inside a namespace-carrying wrapper (error #9)."""
    decls = "".join(' xmlns:%s="%s"' % (p, u)
                    for p, u in sorted(nsmap.items()))
    wrapped = "<vdi-wrap%s>%s</vdi-wrap>" % (decls, op.xml)
    try:
        wrapper = etree.fromstring(wrapped.encode("utf-8"))
    except etree.XMLSyntaxError as e:
        raise _err("fragment failed to parse: %s" % e, source=op.source)
    children = list(wrapper)
    if len(children) != 1:
        raise _err("fragment must contain exactly one root element, "
                   "found %d" % len(children), source=op.source)
    frag = children[0]
    frag.tail = None
    return frag


def _apply_op(tree, op, nsmap):
    """Dispatch one PatchOp against the live tree (errors #8-9)."""
    nodes = _xpath_elements(tree, op.xpath, nsmap, op.source)

    if op.op == "upsert":
        if len(nodes) > 1:
            raise _err("xpath '%s' matched %d elements; upsert allows at "
                       "most 1" % (op.xpath, len(nodes)), source=op.source)
        frag = _parse_fragment(op, nsmap)
        if nodes:
            node = nodes[0]
            parent = node.getparent()
            if parent is None:
                raise _err("cannot replace the document root",
                           source=op.source)
            parent.replace(node, frag)
        else:
            parent_xpath = _upsert_parent_xpath(op)
            parents = _xpath_elements(tree, parent_xpath, nsmap, op.source)
            if len(parents) != 1:
                raise _err("upsert parent '%s' matched %d element(s), "
                           "requires exactly 1"
                           % (parent_xpath, len(parents)), source=op.source)
            parents[0].append(frag)
        return

    _check_cardinality(op, nodes)
    frag = _parse_fragment(op, nsmap) if op.op in FRAGMENT_OPS else None

    for node in nodes:
        new = copy.deepcopy(frag) if frag is not None else None
        parent = node.getparent()
        if op.op in ("replace", "remove", "insert-before", "insert-after"):
            if parent is None:
                raise _err("op '%s' cannot target the document root"
                           % op.op, source=op.source)
        if op.op == "replace":
            parent.replace(node, new)
        elif op.op == "remove":
            parent.remove(node)
        elif op.op == "append-xml":
            node.append(new)
        elif op.op == "prepend-xml":
            node.insert(0, new)
        elif op.op == "insert-before":
            node.addprevious(new)
        elif op.op == "insert-after":
            node.addnext(new)
        elif op.op == "set-attr":
            node.set(op.name, op.value)
        elif op.op == "remove-attr":
            node.attrib.pop(op.name, None)
        elif op.op == "set-text":
            node.text = op.value


def _serialize(tree, nsmap):
    """Hoist namespaces, re-indent, serialize, re-parse (error #10)."""
    etree.cleanup_namespaces(tree, top_nsmap=nsmap)
    etree.indent(tree, space="  ")
    final = etree.tostring(tree, encoding="unicode")
    try:
        etree.fromstring(final.encode("utf-8"))
    except etree.XMLSyntaxError as e:
        raise _err("final document failed re-parse: %s" % e)
    return final


def _log_diff(instance, before, after):
    diff = "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile="base", tofile="override"))
    LOG.info("vdi xml_override diff:\n%s", diff, instance=instance)


# -- operator dry-run CLI (design 9.8) ----------------------------------

def _static_context_source(specs, context_file):
    """Build a synthetic Jinja context for offline rendering."""
    data = {}
    if context_file:
        with open(context_file) as f:
            data = yaml.safe_load(f) or {}

    def context_builder(base_tree):
        instance = {"uuid": "00000000-0000-0000-0000-000000000000",
                    "name": "dry-run", "hostname": "dry-run",
                    "os_type": "windows", "vcpus": 4, "memory_mb": 8192}
        instance.update(data.get("instance") or {})
        flavor = {"name": "dry-run", "flavorid": "dry-run",
                  "vcpus": instance["vcpus"],
                  "memory_mb": instance["memory_mb"]}
        flavor.update(data.get("flavor") or {})
        flavor["extra_specs"] = dict(specs)
        image = {"image_name": None, "image_id": None}
        image.update(data.get("image") or {})

        def flag(name, default=""):
            return specs.get("vdi:flag." + name, default)

        def base(xpath):
            results = etree.XPath(xpath, namespaces=NSMAP)(base_tree)
            if not isinstance(results, list):
                return [str(results)]
            return [r.text or "" if isinstance(r, etree._Element)
                    else str(r) for r in results]

        return {"instance": instance, "flavor": flavor,
                "specs": flavor["extra_specs"], "flag": flag,
                "base": base, "image": image,
                "host": data.get("host", "dry-run-host")}

    return context_builder


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m nova.virt.libvirt.xml_override",
        description="Dry-run the VDI XML override engine offline.")
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render", help="render overrides against a "
                                           "captured base domain XML")
    render.add_argument("--base", required=True,
                        help="file containing the base domain XML")
    render.add_argument("--spec", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="extra spec, repeatable (e.g. "
                             "vdi:profile=latency-gold)")
    render.add_argument("--context", default=None,
                        help="YAML file overriding the synthetic Jinja "
                             "context (instance/flavor/image/host keys)")
    render.add_argument("--profile-dir", default=None,
                        help="profile directory (default: "
                             "[vdi]profile_dir)")
    render.add_argument("--diff", action="store_true",
                        help="print a unified diff instead of the XML")
    args = parser.parse_args(argv)

    specs = {}
    for item in args.spec:
        if "=" not in item:
            print("--spec must be KEY=VALUE: %r" % item, file=sys.stderr)
            return 2
        key, _, value = item.partition("=")
        specs[key] = value

    with open(args.base) as f:
        base_xml = f.read()

    try:
        vdi_keys = _collect_vdi_keys(specs)
        final = _render(
            base_xml, vdi_keys,
            _static_context_source(specs, args.context),
            profile_dir=args.profile_dir or CONF.vdi.profile_dir,
            max_ops=CONF.vdi.max_ops,
            max_fragment_bytes=CONF.vdi.max_fragment_bytes,
            extra_denied=tuple(CONF.vdi.extra_denied_xpaths))
    except VDIXMLOverrideError as e:
        print(str(e), file=sys.stderr)
        return 2

    if args.diff:
        sys.stdout.writelines(difflib.unified_diff(
            base_xml.splitlines(keepends=True),
            final.splitlines(keepends=True),
            fromfile="base", tofile="override"))
    else:
        print(final)
    return 0


if __name__ == "__main__":
    sys.exit(main())

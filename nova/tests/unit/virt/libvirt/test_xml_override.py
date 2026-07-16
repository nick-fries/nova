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

"""Unit tests for nova.virt.libvirt.xml_override (VDI XML overrides)."""

import io
import os
from unittest import mock

import fixtures
from lxml import etree
from oslo_utils.fixture import uuidsentinel as uuids

from nova import context as nova_context
from nova import objects
from nova import test
from nova.tests.unit import fake_instance
from nova.virt.libvirt import xml_override

BASE_XML = """<domain type="kvm">
  <name>instance-00000001</name>
  <uuid>e375f556-b3bd-4d34-a3af-c0d476b53fb6</uuid>
  <memory unit="KiB">4194304</memory>
  <vcpu placement="static">2</vcpu>
  <os>
    <type arch="x86_64" machine="q35">hvm</type>
  </os>
  <features>
    <acpi/>
    <apic/>
    <hyperv mode="custom">
      <relaxed state="on"/>
      <vapic state="on"/>
      <spinlocks state="on" retries="8191"/>
    </hyperv>
    <vmcoreinfo/>
  </features>
  <cpu mode="host-model"/>
  <clock offset="localtime">
    <timer name="pit" tickpolicy="delay"/>
    <timer name="rtc" tickpolicy="catchup"/>
    <timer name="hypervclock" present="yes"/>
  </clock>
  <devices>
    <disk type="file" device="disk"/>
    <interface type="bridge"/>
    <graphics type="spice"/>
  </devices>
</domain>"""

NSMAP = {"qemu": xml_override.QEMU_NS}


class _Base(test.NoDBTestCase):

    def setUp(self):
        super().setUp()
        self.context = nova_context.get_admin_context()
        self.image_meta = objects.ImageMeta.from_dict({
            "id": uuids.image_id,
            "name": "win10-gold",
            "properties": {"hw_machine_type": "q35",
                           "os_type": "windows"},
        })

    def _instance(self, specs):
        flavor = objects.Flavor(
            name="vdi.test", flavorid="vdi-test", vcpus=2,
            memory_mb=4096, root_gb=10, ephemeral_gb=0, swap=0,
            extra_specs=specs)
        return fake_instance.fake_instance_obj(
            self.context, flavor=flavor, display_name="vm1",
            hostname="vm1", os_type="windows", vcpus=2, memory_mb=4096)

    def _apply(self, specs, base=BASE_XML, image_meta=None):
        return xml_override.apply(
            base, self._instance(specs), image_meta or self.image_meta)

    def _apply_tree(self, specs, **kwargs):
        return etree.fromstring(
            self._apply(specs, **kwargs).encode("utf-8"))

    def _assert_fails(self, specs, *needles, **kwargs):
        ex = self.assertRaises(
            xml_override.VDIXMLOverrideError, self._apply, specs, **kwargs)
        for needle in needles:
            self.assertIn(needle, str(ex))
        return ex


class TestFastPathAndKillSwitch(_Base):

    def test_no_vdi_keys_returns_same_object(self):
        # No parse, no copy: identity, not just equality.
        out = self._apply({"hw:cpu_policy": "dedicated"})
        self.assertIs(BASE_XML, out)

    def test_empty_specs_returns_same_object(self):
        self.assertIs(BASE_XML, self._apply({}))

    def test_kill_switch_returns_base_with_warning(self):
        self.flags(enabled=False, group="vdi")
        with mock.patch.object(xml_override.LOG, "warning") as m_warn:
            out = self._apply(
                {"vdi:xml.1": "/domain/cpu|set-attr:mode|host-passthrough"})
        self.assertIs(BASE_XML, out)
        m_warn.assert_called_once()

    def test_unknown_vdi_key_fails(self):
        self._assert_fails({"vdi:porfile": "gold"}, "vdi:porfile")

    def test_flag_only_no_ops_fails(self):
        self._assert_fails({"vdi:flag.evmcs": "true"}, "no ops resolved")


class TestMicroOpGrammar(_Base):

    def test_wrong_field_count(self):
        self._assert_fails({"vdi:xml.1": "/domain/cpu|remove"},
                           "flavor:vdi:xml.1", "exactly two unescaped")

    def test_escaped_pipe_in_payload(self):
        tree = self._apply_tree(
            {"vdi:xml.1": "/domain/cpu|set-attr:check|a\\|b"})
        self.assertEqual("a|b", tree.xpath("/domain/cpu/@check")[0])

    def test_unknown_op(self):
        self._assert_fails({"vdi:xml.1": "/domain/cpu|frobnicate|x"},
                           "unknown op 'frobnicate'")

    def test_unknown_match_mode(self):
        self._assert_fails({"vdi:xml.1": "/domain/cpu|remove@maybe|"},
                           "unknown match mode 'maybe'")

    def test_attr_op_requires_name(self):
        self._assert_fails({"vdi:xml.1": "/domain/cpu|set-attr|x"},
                           "requires an attribute name")

    def test_arg_rejected_on_non_attr_op(self):
        self._assert_fails({"vdi:xml.1": "/domain/cpu|remove:foo|"},
                           "does not take")

    def test_payload_rejected_on_remove(self):
        self._assert_fails({"vdi:xml.1": "/domain/features/acpi|remove|x"},
                           "empty payload")

    def test_empty_xpath(self):
        self._assert_fails({"vdi:xml.1": "|remove|"}, "empty xpath")

    def test_duplicate_nn_after_normalization(self):
        self._assert_fails(
            {"vdi:xml.2": "/domain/cpu|set-attr:a|1",
             "vdi:xml.02": "/domain/cpu|set-attr:b|2"},
            "duplicate micro-op number 2")

    def test_nn_out_of_range(self):
        self._assert_fails({"vdi:xml.0": "/domain/cpu|set-attr:a|1"},
                           "must be 1-99")
        self._assert_fails({"vdi:xml.100": "/domain/cpu|set-attr:a|1"},
                           "must be 1-99")

    def test_numeric_ordering_2_before_10(self):
        # NN=2 sets the attr, NN=10 overwrites it: last (numeric) wins.
        tree = self._apply_tree(
            {"vdi:xml.10": "/domain/cpu|set-attr:mode|second",
             "vdi:xml.2": "/domain/cpu|set-attr:mode|first"})
        self.assertEqual("second", tree.xpath("/domain/cpu/@mode")[0])


class TestOps(_Base):

    def test_replace_whole_block_preserves_position(self):
        tree = self._apply_tree({
            "vdi:xml.1": "/domain/features/hyperv|replace|"
                         '<hyperv mode="custom"><stimer state="on">'
                         '<direct state="on"/></stimer></hyperv>'})
        features = tree.xpath("/domain/features")[0]
        names = [c.tag for c in features]
        # hyperv stays third, between apic and vmcoreinfo.
        self.assertEqual(["acpi", "apic", "hyperv", "vmcoreinfo"], names)
        self.assertEqual(
            "on", tree.xpath("/domain/features/hyperv/stimer/direct"
                             "/@state")[0])
        # Old children are gone with the parent.
        self.assertEqual([], tree.xpath("/domain/features/hyperv/relaxed"))

    def test_remove(self):
        tree = self._apply_tree(
            {"vdi:xml.1": "/domain/features/vmcoreinfo|remove|"})
        self.assertEqual([], tree.xpath("/domain/features/vmcoreinfo"))

    def test_append_xml(self):
        tree = self._apply_tree(
            {"vdi:xml.1": "/domain/clock|append-xml|"
                          '<timer name="hpet" present="no"/>'})
        timers = tree.xpath("/domain/clock/timer")
        self.assertEqual("hpet", timers[-1].get("name"))

    def test_prepend_xml(self):
        tree = self._apply_tree(
            {"vdi:xml.1": "/domain/clock|prepend-xml|"
                          '<timer name="hpet" present="no"/>'})
        timers = tree.xpath("/domain/clock/timer")
        self.assertEqual("hpet", timers[0].get("name"))

    def test_insert_before_and_after(self):
        tree = self._apply_tree({
            "vdi:xml.1": '/domain/clock/timer[@name="rtc"]|insert-before|'
                         '<timer name="kvmclock" present="no"/>',
            "vdi:xml.2": '/domain/clock/timer[@name="rtc"]|insert-after|'
                         '<timer name="tsc" mode="native"/>'})
        names = [t.get("name")
                 for t in tree.xpath("/domain/clock/timer")]
        self.assertEqual(
            ["pit", "kvmclock", "rtc", "tsc", "hypervclock"], names)

    def test_upsert_existing_replaces(self):
        tree = self._apply_tree(
            {"vdi:xml.1": "/domain/features/hyperv/spinlocks|upsert|"
                          '<spinlocks state="on" retries="4095"/>'})
        self.assertEqual(
            ["4095"],
            tree.xpath("/domain/features/hyperv/spinlocks/@retries"))

    def test_upsert_missing_appends_to_parent(self):
        tree = self._apply_tree(
            {"vdi:xml.1": "/domain/features/hyperv/stimer|upsert|"
                          '<stimer state="on"/>'})
        hyperv = tree.xpath("/domain/features/hyperv")[0]
        self.assertEqual("stimer", hyperv[-1].tag)

    def test_upsert_with_predicate_xpath(self):
        tree = self._apply_tree(
            {"vdi:xml.1": '/domain/clock/timer[@name="hpet"]|upsert|'
                          '<timer name="hpet" present="yes"/>'})
        self.assertEqual(
            ["yes"],
            tree.xpath('/domain/clock/timer[@name="hpet"]/@present'))

    def test_upsert_parent_missing_fails(self):
        self._assert_fails(
            {"vdi:xml.1": "/domain/pm/suspend-to-mem|upsert|"
                          '<suspend-to-mem enabled="no"/>'},
            "upsert parent", "matched 0")

    def test_upsert_root_fails(self):
        self._assert_fails(
            {"vdi:xml.1": "/domain|upsert|<domain/>"},
            "cannot upsert the document root")

    def test_set_attr_creates_and_overwrites(self):
        tree = self._apply_tree({
            "vdi:xml.1": "/domain/cpu|set-attr:mode|host-passthrough",
            "vdi:xml.2": "/domain/cpu|set-attr:check|none"})
        cpu = tree.xpath("/domain/cpu")[0]
        self.assertEqual("host-passthrough", cpu.get("mode"))
        self.assertEqual("none", cpu.get("check"))

    def test_remove_attr_absent_is_ok(self):
        tree = self._apply_tree({
            "vdi:xml.1": "/domain/cpu|remove-attr:mode|",
            "vdi:xml.2": "/domain/cpu|remove-attr:nonexistent|"})
        self.assertIsNone(tree.xpath("/domain/cpu")[0].get("mode"))

    def test_set_text_and_empty_clears(self):
        tree = self._apply_tree(
            {"vdi:xml.1": "/domain/os/type|set-text|linux"})
        self.assertEqual("linux", tree.xpath("/domain/os/type")[0].text)
        tree = self._apply_tree(
            {"vdi:xml.1": "/domain/os/type|set-text|"})
        self.assertFalse((tree.xpath("/domain/os/type")[0].text or
                          "").strip())

    def test_qemu_commandline_namespace_hoisted(self):
        out = self._apply(
            {"vdi:xml.1": "/domain|append-xml|<qemu:commandline>"
                          '<qemu:arg value="-overcommit"/>'
                          "</qemu:commandline>"})
        self.assertIn('xmlns:qemu="%s"' % xml_override.QEMU_NS,
                      out.splitlines()[0])
        tree = etree.fromstring(out.encode("utf-8"))
        args = tree.xpath("/domain/qemu:commandline/qemu:arg/@value",
                          namespaces=NSMAP)
        self.assertEqual(["-overcommit"], args)

    def test_determinism(self):
        specs = {
            "vdi:xml.1": "/domain/features/hyperv/stimer|upsert|"
                         '<stimer state="on"/>',
            "vdi:xml.2": "/domain|append-xml|<qemu:commandline>"
                         '<qemu:arg value="-overcommit"/>'
                         "</qemu:commandline>"}
        self.assertEqual(self._apply(specs), self._apply(specs))

    def test_output_reparses_and_base_untouched(self):
        out = self._apply(
            {"vdi:xml.1": "/domain/features/acpi|remove|"})
        etree.fromstring(out.encode("utf-8"))
        # The input string itself must be untouched.
        self.assertIn("<acpi/>", BASE_XML)


class TestCardinality(_Base):

    def test_one_zero_matches_fails(self):
        self._assert_fails(
            {"vdi:xml.1": "/domain/pm|set-attr:x|y"},
            "matched 0", "match=one", "flavor:vdi:xml.1")

    def test_one_multiple_matches_fails(self):
        self._assert_fails(
            {"vdi:xml.1": "/domain/clock/timer|set-attr:x|y"},
            "matched 3", "match=one")

    def test_all_applies_to_each(self):
        tree = self._apply_tree(
            {"vdi:xml.1": "/domain/clock/timer|set-attr@all:marked|yes"})
        self.assertEqual(
            ["yes"] * 3, tree.xpath("/domain/clock/timer/@marked"))

    def test_all_zero_matches_fails(self):
        self._assert_fails(
            {"vdi:xml.1": "/domain/pm|set-attr@all:x|y"},
            "matched 0", "match=all")

    def test_opt_zero_is_silent_noop(self):
        out = self._apply(
            {"vdi:xml.1": "/domain/features/pae|remove@opt|",
             "vdi:xml.2": "/domain/cpu|set-attr:check|none"})
        tree = etree.fromstring(out.encode("utf-8"))
        self.assertEqual(["none"], tree.xpath("/domain/cpu/@check"))

    def test_opt_multiple_matches_fails(self):
        self._assert_fails(
            {"vdi:xml.1": "/domain/clock/timer|remove@opt|"},
            "matched 3", "match=opt")

    def test_any_zero_and_many_never_fail(self):
        tree = self._apply_tree(
            {"vdi:xml.1": "/domain/features/pae|remove@any|",
             "vdi:xml.2": "/domain/clock/timer|set-attr@any:m|1"})
        self.assertEqual(["1"] * 3, tree.xpath("/domain/clock/timer/@m"))

    def test_fragment_applied_to_multiple_nodes_is_copied(self):
        tree = self._apply_tree(
            {"vdi:xml.1": "/domain/clock/timer|append-xml@all|"
                          "<child/>"})
        self.assertEqual(3, len(tree.xpath("/domain/clock/timer/child")))


class TestErrorsAndLimits(_Base):

    def test_bad_xpath_syntax(self):
        self._assert_fails(
            {"vdi:xml.1": "/domain[[[|remove|"}, "invalid xpath")

    def test_xpath_selecting_attributes_fails(self):
        self._assert_fails(
            {"vdi:xml.1": "/domain/@type|set-text|x"},
            "must select elements")

    def test_unparsable_fragment(self):
        self._assert_fails(
            {"vdi:xml.1": "/domain|append-xml|<unclosed"},
            "fragment failed to parse")

    def test_fragment_with_two_roots(self):
        self._assert_fails(
            {"vdi:xml.1": "/domain|append-xml|<a/><b/>"},
            "exactly one root element")

    def test_fragment_size_limit(self):
        self.flags(max_fragment_bytes=8, group="vdi")
        self._assert_fails(
            {"vdi:xml.1": "/domain|append-xml|<toolongfragment/>"},
            "max_fragment_bytes")

    def test_max_ops_limit(self):
        self.flags(max_ops=1, group="vdi")
        self._assert_fails(
            {"vdi:xml.1": "/domain/cpu|set-attr:a|1",
             "vdi:xml.2": "/domain/cpu|set-attr:b|2"},
            "exceeds [vdi]max_ops=1")

    def test_remove_root_fails(self):
        self._assert_fails(
            {"vdi:xml.1": "/domain|remove|"},
            "cannot target the document root")

    def test_base_xml_unparsable(self):
        self._assert_fails(
            {"vdi:xml.1": "/domain/cpu|set-attr:a|1"},
            "base XML failed to parse", base="<broken")


class TestDenyList(_Base):

    def test_denied_prefixes(self):
        for xpath in ("/domain/name", "/domain/uuid", "/domain/memory",
                      "/domain/vcpu", "/domain/numatune",
                      "/domain/cpu/numa/cell",
                      "/domain/memoryBacking/hugepages",
                      "/domain/devices/disk",
                      "/domain/devices/interface"):
            self._assert_fails(
                {"vdi:xml.1": "%s|remove@any|" % xpath},
                "denied region")

    def test_predicates_stripped_before_check(self):
        self._assert_fails(
            {"vdi:xml.1": '/domain/devices/disk[@device="disk"]|'
                          "remove@any|"},
            "denied region", "/domain/devices/disk")

    def test_prefix_boundary_not_overbroad(self):
        # /domain/memoryBacking is allowed even though /domain/memory
        # is denied; /domain/vcpus (hypothetical) vs /domain/vcpu ditto.
        tree = self._apply_tree(
            {"vdi:xml.1": "/domain/memoryBacking|upsert|"
                          "<memoryBacking><source type='memfd'/>"
                          "</memoryBacking>"})
        self.assertEqual(
            ["memfd"],
            tree.xpath("/domain/memoryBacking/source/@type"))

    def test_features_clock_qemu_are_allowed(self):
        tree = self._apply_tree({
            "vdi:xml.1": "/domain/features/hyperv/stimer|upsert|"
                         '<stimer state="on"/>',
            "vdi:xml.2": '/domain/clock/timer[@name="hpet"]|upsert|'
                         '<timer name="hpet" present="no"/>',
            "vdi:xml.3": "/domain|append-xml|<qemu:commandline>"
                         '<qemu:arg value="-overcommit"/>'
                         "</qemu:commandline>"})
        self.assertTrue(tree.xpath("/domain/features/hyperv/stimer"))

    def test_extra_denied_xpaths_config(self):
        self.flags(extra_denied_xpaths=["/domain/os"], group="vdi")
        self._assert_fails(
            {"vdi:xml.1": "/domain/os/type|set-attr:machine|pc"},
            "denied region", "/domain/os")

    def test_upsert_into_denied_parent_fails(self):
        self._assert_fails(
            {"vdi:xml.1": "/domain/devices/disk/driver|upsert|"
                          '<driver io="native"/>'},
            "denied region")


class TestProfiles(_Base):

    def setUp(self):
        super().setUp()
        self.profile_dir = self.useFixture(fixtures.TempDir()).path
        self.flags(profile_dir=self.profile_dir, group="vdi")

    def _write_profile(self, name, content, suffix=".yaml.j2"):
        path = os.path.join(self.profile_dir, name + suffix)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_profile_basic(self):
        self._write_profile("gold", """
description: test profile
ops:
  - xpath: /domain/features/hyperv/stimer
    op: upsert
    xml: <stimer state="on"/>
  - xpath: /domain/cpu
    op: set-attr
    name: mode
    value: host-passthrough
""")
        tree = self._apply_tree({"vdi:profile": "gold"})
        self.assertTrue(tree.xpath("/domain/features/hyperv/stimer"))
        self.assertEqual(
            ["host-passthrough"], tree.xpath("/domain/cpu/@mode"))

    def test_plain_yaml_suffix_also_found(self):
        self._write_profile("plain", """
ops:
  - {xpath: /domain/cpu, op: set-attr, name: check, value: none}
""", suffix=".yaml")
        tree = self._apply_tree({"vdi:profile": "plain"})
        self.assertEqual(["none"], tree.xpath("/domain/cpu/@check"))

    def test_missing_profile_fails(self):
        self._assert_fails({"vdi:profile": "nope"},
                           "profile 'nope' not found", "profile:nope")

    def test_profile_name_traversal_rejected(self):
        self._assert_fails({"vdi:profile": "../../etc/passwd"},
                           "invalid profile name")
        self._assert_fails({"vdi:profile": "_partials/x"},
                           "invalid profile name")

    def test_jinja_undefined_variable_fails(self):
        self._write_profile("bad", """
ops:
  - xpath: /domain/cpu
    op: set-attr
    name: mode
    value: {{ no_such_variable }}
""")
        self._assert_fails({"vdi:profile": "bad"},
                           "template error", "profile:bad")

    def test_jinja_syntax_error_fails(self):
        self._write_profile("bad", "ops:\n{% if %}\n")
        self._assert_fails({"vdi:profile": "bad"}, "template error")

    def test_invalid_yaml_fails(self):
        self._write_profile("bad", "ops: [unclosed\n")
        self._assert_fails({"vdi:profile": "bad"}, "not valid YAML")

    def test_unknown_top_level_key_fails(self):
        self._write_profile("bad", """
opps:
  - {xpath: /domain, op: remove}
""")
        self._assert_fails({"vdi:profile": "bad"},
                           "unknown top-level key")

    def test_unknown_op_key_typo_fails(self):
        self._write_profile("bad", """
ops:
  - xpath: /domain/cpu
    op: set-attr
    name: mode
    vlaue: host-passthrough
""")
        self._assert_fails({"vdi:profile": "bad"},
                           "unknown op key", "vlaue", "profile:bad[0]")

    def test_empty_ops_fails(self):
        self._write_profile("bad", "ops: []\n")
        self._assert_fails({"vdi:profile": "bad"}, "non-empty list")

    def test_profile_op_error_carries_provenance(self):
        self._write_profile("gold", """
ops:
  - {xpath: /domain/cpu, op: set-attr, name: a, value: '1'}
  - {xpath: /domain/nonexistent, op: remove}
""")
        self._assert_fails({"vdi:profile": "gold"},
                           "profile:gold[1]", "matched 0")

    def test_profile_chain_and_microop_ordering(self):
        # base profile installs the block, tier profile refines it,
        # micro-op applies last and wins.
        self._write_profile("base", """
ops:
  - xpath: /domain/features/hyperv
    op: replace
    xml: |
      <hyperv><spinlocks state="on" retries="8191"/></hyperv>
""")
        self._write_profile("tier", """
ops:
  - xpath: /domain/features/hyperv/spinlocks
    op: set-attr
    name: retries
    value: '4095'
""")
        tree = self._apply_tree({
            "vdi:profile": "base,tier",
            "vdi:xml.1": "/domain/features/hyperv/spinlocks|"
                         "set-attr:retries|2047"})
        self.assertEqual(
            ["2047"],
            tree.xpath("/domain/features/hyperv/spinlocks/@retries"))

    def test_flag_helper_conditional(self):
        self._write_profile("gold", """
ops:
  - xpath: /domain/features/hyperv
    op: replace
    xml: |
      <hyperv>
        <stimer state="on"/>
        {% if flag('evmcs') == 'true' %}
        <evmcs state="on"/>
        {% endif %}
      </hyperv>
""")
        tree = self._apply_tree({"vdi:profile": "gold"})
        self.assertEqual([], tree.xpath("/domain/features/hyperv/evmcs"))
        tree = self._apply_tree({"vdi:profile": "gold",
                                 "vdi:flag.evmcs": "true"})
        self.assertTrue(tree.xpath("/domain/features/hyperv/evmcs"))

    def test_base_helper_queries_pristine_xml(self):
        self._write_profile("gold", """
ops:
{% if base('/domain/features/hyperv') %}
  - xpath: /domain/features/hyperv
    op: set-attr
    name: seen
    value: 'yes'
{% else %}
  - xpath: /domain/features
    op: append-xml
    xml: <hyperv/>
{% endif %}
""")
        tree = self._apply_tree({"vdi:profile": "gold"})
        self.assertEqual(
            ["yes"], tree.xpath("/domain/features/hyperv/@seen"))

    def test_context_instance_flavor_image_host(self):
        self.flags(host="cmp-01")
        self._write_profile("ctx", """
ops:
  - xpath: /domain/features/hyperv
    op: set-attr
    name: note
    value: "{{ instance.name }}/{{ flavor.name }}/\
{{ image['hw_machine_type'] }}/{{ host }}/{{ specs['vdi:profile'] }}"
""")
        tree = self._apply_tree({"vdi:profile": "ctx"})
        self.assertEqual(
            ["vm1/vdi.test/q35/cmp-01/ctx"],
            tree.xpath("/domain/features/hyperv/@note"))

    def test_profile_extra_namespaces(self):
        self._write_profile("ns", """
namespaces:
  foo: http://example.com/foo/1.0
ops:
  - xpath: /domain
    op: append-xml
    xml: <foo:widget xmlns:ignored="x"/>
""")
        tree = self._apply_tree({"vdi:profile": "ns"})
        self.assertEqual(1, len(tree.xpath(
            "/domain/foo:widget",
            namespaces={"foo": "http://example.com/foo/1.0"})))

    def test_namespace_prefix_conflict_fails(self):
        self._write_profile("ns", """
namespaces:
  qemu: http://example.com/not-qemu
ops:
  - {xpath: /domain/cpu, op: set-attr, name: a, value: '1'}
""")
        self._assert_fails({"vdi:profile": "ns"},
                           "redefined with a different URI")

    def test_sandbox_blocks_dangerous_template(self):
        self._write_profile("evil", """
ops:
  - xpath: /domain/cpu
    op: set-attr
    name: a
    value: "{{ ''.__class__.__mro__ }}"
""")
        self._assert_fails({"vdi:profile": "evil"}, "template error")


class TestDryRunCLI(_Base):

    def setUp(self):
        super().setUp()
        self.profile_dir = self.useFixture(fixtures.TempDir()).path
        base_dir = self.useFixture(fixtures.TempDir()).path
        self.base_file = os.path.join(base_dir, "base.xml")
        with open(self.base_file, "w") as f:
            f.write(BASE_XML)

    def _run(self, *argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch("sys.stdout", stdout), \
                mock.patch("sys.stderr", stderr):
            rc = xml_override.main(list(argv))
        return rc, stdout.getvalue(), stderr.getvalue()

    def test_render_micro_op(self):
        rc, out, _ = self._run(
            "render", "--base", self.base_file,
            "--spec", "vdi:xml.1=/domain/features/hyperv/stimer|upsert|"
                      '<stimer state="on"/>')
        self.assertEqual(0, rc)
        self.assertIn("<stimer state=\"on\"/>", out)

    def test_render_diff_mode(self):
        rc, out, _ = self._run(
            "render", "--base", self.base_file, "--diff",
            "--spec", "vdi:xml.1=/domain/cpu|set-attr:check|none")
        self.assertEqual(0, rc)
        self.assertIn("+", out)
        self.assertIn("check=\"none\"", out)

    def test_render_profile_with_flag(self):
        with open(os.path.join(self.profile_dir, "g.yaml.j2"), "w") as f:
            f.write("ops:\n"
                    "  - xpath: /domain/cpu\n"
                    "    op: set-attr\n"
                    "    name: note\n"
                    "    value: \"{{ flag('x', 'def') }}\"\n")
        rc, out, _ = self._run(
            "render", "--base", self.base_file,
            "--profile-dir", self.profile_dir,
            "--spec", "vdi:profile=g", "--spec", "vdi:flag.x=hello")
        self.assertEqual(0, rc)
        self.assertIn('note="hello"', out)

    def test_error_exits_nonzero_with_message(self):
        rc, _, err = self._run(
            "render", "--base", self.base_file,
            "--profile-dir", self.profile_dir,
            "--spec", "vdi:profile=missing")
        self.assertEqual(2, rc)
        self.assertIn("not found", err)

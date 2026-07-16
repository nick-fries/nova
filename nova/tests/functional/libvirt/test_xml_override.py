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

"""Functional tests for VDI XML overrides through the real libvirt driver.

Boots servers via the API against fakelibvirt with ``vdi:`` flavors and
asserts on the domain XML the driver actually defined.  These are the
rebase early-warning tests (design doc section 11.2).
"""

import copy
import os

import fixtures
from lxml import etree
from oslo_utils.fixture import uuidsentinel as uuids

from nova.tests.functional.libvirt import base
from nova.virt.libvirt import driver as libvirt_driver
from nova.virt.libvirt import xml_override

QEMU_NS = xml_override.QEMU_NS

PROFILE = """
description: functional-test latency tier
ops:
  - xpath: /domain/features/hyperv
    op: replace
    xml: |
      <hyperv mode="custom">
        <relaxed state="on"/>
        <vapic state="on"/>
        <spinlocks state="on" retries="8191"/>
        <vpindex state="on"/>
        <runtime state="on"/>
        <synic state="on"/>
        <stimer state="on">
          <direct state="on"/>
        </stimer>
        <reset state="on"/>
        <frequencies state="on"/>
        <reenlightenment state="on"/>
        <tlbflush state="on"/>
        <ipi state="on"/>
        {% if flag('evmcs') == 'true' %}
        <evmcs state="on"/>
        {% endif %}
      </hyperv>
  - xpath: /domain/clock/timer[@name="hypervclock"]
    op: upsert
    xml: <timer name="hypervclock" present="yes"/>
  - xpath: /domain
    op: append-xml
    xml: |
      <qemu:commandline>
        <qemu:arg value="-overcommit"/>
        <qemu:arg value="cpu-pm=on"/>
      </qemu:commandline>
"""


class VDIXMLOverrideTest(base.ServersTestBase):

    microversion = 'latest'
    ADMIN_API = True

    def setUp(self):
        super().setUp()
        self.profile_dir = self.useFixture(fixtures.TempDir()).path
        self.flags(profile_dir=self.profile_dir, group='vdi')
        with open(os.path.join(self.profile_dir,
                               'latency-gold.yaml.j2'), 'w') as f:
            f.write(PROFILE)

        # Spy on the exact XML _get_guest_xml returns (post-override).
        # fakelibvirt's Domain.XMLDesc re-serializes from parsed state
        # and drops <features>/<clock>/qemu:commandline, so asserting on
        # get_xml_desc() would test the fake, not the driver.
        self.rendered_xml = []
        real = libvirt_driver.LibvirtDriver._get_guest_xml

        def _spy(driver_self, *args, **kwargs):
            xml = real(driver_self, *args, **kwargs)
            self.rendered_xml.append(xml)
            return xml

        self.useFixture(fixtures.MonkeyPatch(
            'nova.virt.libvirt.driver.LibvirtDriver._get_guest_xml',
            _spy))

        self.compute = self.start_compute('host1')

        # A windows image, as every real VDI guest would use.
        image = copy.deepcopy(self.glance.image1)
        image['id'] = uuids.windows_image
        image['properties']['os_type'] = 'windows'
        self.glance.create(None, image)

    def _boot(self, extra_spec, expected_state='ACTIVE'):
        flavor_id = self._create_flavor(extra_spec=extra_spec)
        return self._create_server(
            flavor_id=flavor_id, image_uuid=uuids.windows_image,
            networks='none', expected_state=expected_state)

    def _last_tree(self):
        self.assertTrue(self.rendered_xml,
                        '_get_guest_xml was never reached')
        xml = self.rendered_xml[-1]
        return etree.fromstring(xml.encode('utf-8')), xml

    def test_boot_with_profile_and_micro_op(self):
        self._boot({
            'vdi:profile': 'latency-gold',
            'vdi:flag.evmcs': 'true',
            'vdi:xml.1': '/domain/features/hyperv/spinlocks|'
                         'set-attr:retries|4095',
        })
        tree, xml = self._last_tree()

        # Profile: full enlightenment replacement, incl. what stock nova
        # cannot emit (stimer/direct, reenlightenment, evmcs).
        hyperv = tree.xpath('/domain/features/hyperv')
        self.assertEqual(1, len(hyperv))
        self.assertEqual(
            ['on'],
            tree.xpath('/domain/features/hyperv/stimer/direct/@state'))
        self.assertEqual(
            ['on'],
            tree.xpath('/domain/features/hyperv/reenlightenment/@state'))
        self.assertEqual(
            ['on'], tree.xpath('/domain/features/hyperv/evmcs/@state'))
        # Micro-op layered after the profile refined the same block.
        self.assertEqual(
            ['4095'],
            tree.xpath('/domain/features/hyperv/spinlocks/@retries'))
        # Upsert converged with the timer nova already emits for windows.
        self.assertEqual(
            1, len(tree.xpath('/domain/clock/timer[@name="hypervclock"]')))
        # qemu:commandline injected in the proper namespace.
        args = tree.xpath('/domain/qemu:commandline/qemu:arg/@value',
                          namespaces={'qemu': QEMU_NS})
        self.assertEqual(['-overcommit', 'cpu-pm=on'], args)
        self.assertIn('xmlns:qemu="%s"' % QEMU_NS, xml)

    def test_micro_ops_only(self):
        self._boot({
            'vdi:xml.1': '/domain/features/hyperv/stimer|upsert|'
                         '<stimer state="on"/>',
            'vdi:xml.2': '/domain/clock/timer[@name="hpet"]|upsert|'
                         '<timer name="hpet" present="no"/>',
        })
        tree, _ = self._last_tree()
        self.assertEqual(
            ['on'], tree.xpath('/domain/features/hyperv/stimer/@state'))
        self.assertEqual(
            ['no'],
            tree.xpath('/domain/clock/timer[@name="hpet"]/@present'))

    def test_stock_flavor_untouched(self):
        self._boot({})
        tree, xml = self._last_tree()
        # Fast path: no override artifacts anywhere.
        self.assertNotIn('qemu:commandline', xml)
        self.assertEqual([], tree.xpath('/domain/features/hyperv/stimer'))
        # Stock windows enlightenments still present (base behavior).
        self.assertEqual(
            ['on'], tree.xpath('/domain/features/hyperv/relaxed/@state'))

    def test_missing_profile_fails_closed(self):
        # The VDIXMLOverrideError is wrapped in a RescheduledException by
        # the compute manager, so the final instance fault carries the
        # generic retry-exhausted text.  Provenance lands in the logs.
        self._boot({'vdi:profile': 'does-not-exist'},
                   expected_state='ERROR')
        self.assertIn("profile 'does-not-exist' not found",
                      self.stdlog.logger.output)

    def test_denied_xpath_fails_closed(self):
        self._boot({'vdi:xml.1': '/domain/vcpu|set-text|8'},
                   expected_state='ERROR')
        self.assertIn('denied region', self.stdlog.logger.output)

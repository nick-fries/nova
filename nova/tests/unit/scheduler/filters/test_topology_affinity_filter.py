# Copyright 2026 Phase 3 Topology Fork.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""Tests for nova.scheduler.filters.topology_affinity_filter."""

import collections
from unittest import mock

from nova import objects
from nova.scheduler.filters import topology_affinity_filter as taf
from nova import test
from nova.tests.unit.scheduler import fakes


# Helpful named tuple to mimic the SchedulerReportClient's
# ``get_provider_traits`` return shape.
_TraitInfo = collections.namedtuple(
    "_TraitInfo", ["traits", "generation"],
)


def _spec(locality, requested_resources=None):
    """Build a minimal RequestSpec object with a locality extra-spec."""
    extra = {} if locality is None else {"hw:cyborg_locality": locality}
    flavor = objects.Flavor(extra_specs=extra)
    return objects.RequestSpec(
        context=mock.sentinel.ctx,
        flavor=flavor,
    )


def _candidate(mappings, allocations=None):
    """Build a minimal allocation candidate."""
    return {
        "mappings": mappings,
        "allocations": allocations or {},
    }


def _make_report_mock(traits_by_rp, rps_by_uuid):
    """Build a SchedulerReportClient mock that satisfies the filter.

    :param traits_by_rp: dict rp_uuid -> set[str] of traits.
    :param rps_by_uuid: dict rp_uuid -> {'parent_provider_uuid', 'name'}.
    """
    m = mock.MagicMock()

    def _gt(ctx, rp_uuid):
        return _TraitInfo(
            traits=set(traits_by_rp.get(rp_uuid, set())),
            generation=1,
        )

    def _grp(ctx, rp_uuid):
        return rps_by_uuid.get(rp_uuid)

    m.get_provider_traits.side_effect = _gt
    m._get_resource_provider.side_effect = _grp
    return m


class _FilterTestBase(test.NoDBTestCase):
    """Common fixture: build the filter, hand it a mocked report client."""

    def setUp(self):
        super().setUp()
        self.filt = taf.TopologyAffinityFilter()

    def _install_report(self, traits, rps):
        report = _make_report_mock(traits, rps)
        # Stash the report into the filter instance; the filter
        # lazy-init's, but injecting directly avoids importing
        # the real client.
        self.filt._report = report
        return report

    def _host(self, candidates):
        host = fakes.FakeHostState('host1', 'node1', {})
        host.allocation_candidates = candidates
        return host


class TestLocalityNone(_FilterTestBase):
    """hw:cyborg_locality=none / unset is a no-op."""

    def test_locality_none_passes_without_lookups(self):
        spec = _spec("none")
        host = self._host([_candidate({})])
        # No need to install a report client - it must not be called.
        self.filt._report = mock.MagicMock()
        self.assertTrue(self.filt.host_passes(host, spec))
        self.filt._report.get_provider_traits.assert_not_called()

    def test_locality_unset_passes(self):
        spec = _spec(None)
        host = self._host([_candidate({})])
        self.filt._report = mock.MagicMock()
        self.assertTrue(self.filt.host_passes(host, spec))


class TestLocalitySocket(_FilterTestBase):
    """hw:cyborg_locality=socket cases."""

    def test_socket_match_passes(self):
        """Accel under socket_0; NIC tagged CUSTOM_TOPO_SOCKET0."""
        spec = _spec("socket")
        candidate = _candidate(
            mappings={
                "device_profile_0": ["accel-rp"],
                "11111111-1111-1111-1111-111111111111": ["nic-rp"],
            },
        )
        host = self._host([candidate])
        self._install_report(
            traits={
                "accel-rp": {"CUSTOM_AMD_V620_VF"},
                "socket-anchor": {"CUSTOM_SOCKET_ROOT"},
                "host-root": set(),
                "nic-rp": {"CUSTOM_TOPO_SOCKET0",
                           "CUSTOM_PHYSNET_HPC"},
            },
            rps={
                "accel-rp": {"parent_provider_uuid": "socket-anchor",
                             "name": "host1_devA"},
                "socket-anchor": {
                    "parent_provider_uuid": "host-root",
                    "name": "host1_socket_0",
                },
                "host-root": {"parent_provider_uuid": None,
                              "name": "host1"},
                "nic-rp": {"parent_provider_uuid": None,
                           "name": "host1:hpc:0000:31:00.0"},
            },
        )
        self.assertTrue(self.filt.host_passes(host, spec))
        # Candidate survived.
        self.assertEqual(1, len(host.allocation_candidates))

    def test_socket_mismatch_pruned(self):
        """Accel under socket_0; NIC tagged CUSTOM_TOPO_SOCKET1 -> prune."""
        spec = _spec("socket")
        candidate = _candidate(
            mappings={
                "device_profile_0": ["accel-rp"],
                "11111111-1111-1111-1111-111111111111": ["nic-rp"],
            },
        )
        host = self._host([candidate])
        self._install_report(
            traits={
                "accel-rp": set(),
                "socket-anchor": {"CUSTOM_SOCKET_ROOT"},
                "host-root": set(),
                "nic-rp": {"CUSTOM_TOPO_SOCKET1"},
            },
            rps={
                "accel-rp": {"parent_provider_uuid": "socket-anchor",
                             "name": "host1_devA"},
                "socket-anchor": {
                    "parent_provider_uuid": "host-root",
                    "name": "host1_socket_0",
                },
                "host-root": {"parent_provider_uuid": None,
                              "name": "host1"},
                "nic-rp": {"parent_provider_uuid": None,
                           "name": "host1:hpc:0000:31:00.0"},
            },
        )
        self.assertFalse(self.filt.host_passes(host, spec))

    def test_one_socket_host(self):
        """1-socket host: socket_0 only; NIC has CUSTOM_TOPO_SOCKET0."""
        spec = _spec("socket")
        candidate = _candidate(
            mappings={
                "device_profile_0": ["accel-rp"],
                "11111111-1111-1111-1111-111111111111": ["nic-rp"],
            },
        )
        host = self._host([candidate])
        self._install_report(
            traits={
                "accel-rp": set(),
                "socket-anchor": {"CUSTOM_SOCKET_ROOT"},
                "host-root": set(),
                "nic-rp": {"CUSTOM_TOPO_SOCKET0"},
            },
            rps={
                "accel-rp": {"parent_provider_uuid": "numa-anchor",
                             "name": "host1_devA"},
                "numa-anchor": {
                    "parent_provider_uuid": "socket-anchor",
                    "name": "host1_numa_0",
                },
                "socket-anchor": {
                    "parent_provider_uuid": "host-root",
                    "name": "host1_socket_0",
                },
                "host-root": {"parent_provider_uuid": None,
                              "name": "host1"},
                "nic-rp": {"parent_provider_uuid": None,
                           "name": "host1:hpc:0000:31:00.0"},
            },
        )
        # Include NUMA hop above accel-rp in traits.
        self.filt._report.get_provider_traits.side_effect = (
            lambda ctx, u: _TraitInfo(
                traits=set({
                    'accel-rp': set(),
                    'numa-anchor': {"HW_NUMA_ROOT"},
                    'socket-anchor': {"CUSTOM_SOCKET_ROOT"},
                    'host-root': set(),
                    'nic-rp': {"CUSTOM_TOPO_SOCKET0"},
                }.get(u, set())),
                generation=1,
            )
        )
        self.assertTrue(self.filt.host_passes(host, spec))


class TestLocalityNuma(_FilterTestBase):
    """hw:cyborg_locality=numa cases."""

    def test_numa_match_passes(self):
        spec = _spec("numa")
        candidate = _candidate(
            mappings={
                "device_profile_0": ["accel-rp"],
                "11111111-1111-1111-1111-111111111111": ["nic-rp"],
            },
        )
        host = self._host([candidate])
        self._install_report(
            traits={
                "accel-rp": set(),
                "numa-anchor": {"HW_NUMA_ROOT"},
                "socket-anchor": {"CUSTOM_SOCKET_ROOT"},
                "host-root": set(),
                "nic-rp": {"CUSTOM_TOPO_NUMA0", "CUSTOM_TOPO_SOCKET0"},
            },
            rps={
                "accel-rp": {"parent_provider_uuid": "numa-anchor",
                             "name": "host1_devA"},
                "numa-anchor": {
                    "parent_provider_uuid": "socket-anchor",
                    "name": "host1_numa_0",
                },
                "socket-anchor": {
                    "parent_provider_uuid": "host-root",
                    "name": "host1_socket_0",
                },
                "host-root": {"parent_provider_uuid": None,
                              "name": "host1"},
                "nic-rp": {"parent_provider_uuid": None,
                           "name": "host1:hpc:0000:31:00.0"},
            },
        )
        self.assertTrue(self.filt.host_passes(host, spec))

    def test_numa_mismatch_same_socket_pruned(self):
        """Same socket (0) but different NUMA (0 vs 1) -> prune for NUMA."""
        spec = _spec("numa")
        candidate = _candidate(
            mappings={
                "device_profile_0": ["accel-rp"],
                "11111111-1111-1111-1111-111111111111": ["nic-rp"],
            },
        )
        host = self._host([candidate])
        self._install_report(
            traits={
                "accel-rp": set(),
                "numa-anchor": {"HW_NUMA_ROOT"},  # numa_0
                "socket-anchor": {"CUSTOM_SOCKET_ROOT"},
                "host-root": set(),
                "nic-rp": {"CUSTOM_TOPO_NUMA1", "CUSTOM_TOPO_SOCKET0"},
            },
            rps={
                "accel-rp": {"parent_provider_uuid": "numa-anchor",
                             "name": "host1_devA"},
                "numa-anchor": {
                    "parent_provider_uuid": "socket-anchor",
                    "name": "host1_numa_0",
                },
                "socket-anchor": {
                    "parent_provider_uuid": "host-root",
                    "name": "host1_socket_0",
                },
                "host-root": {"parent_provider_uuid": None,
                              "name": "host1"},
                "nic-rp": {"parent_provider_uuid": None,
                           "name": "host1:hpc:0000:31:00.0"},
            },
        )
        self.assertFalse(self.filt.host_passes(host, spec))


class TestMultiCandidate(_FilterTestBase):
    """Multiple candidates: some pass, some pruned."""

    def setUp(self):
        super().setUp()
        # Two NUMA anchors under two socket anchors. Two NICs, one per
        # socket. Two candidates: one cross-socket (prune), one same-
        # socket (pass).
        traits = {
            'accel-rp-A': set(),
            'numa-anchor-0': {"HW_NUMA_ROOT"},
            'numa-anchor-1': {"HW_NUMA_ROOT"},
            'socket-anchor-0': {"CUSTOM_SOCKET_ROOT"},
            'socket-anchor-1': {"CUSTOM_SOCKET_ROOT"},
            'host-root': set(),
            'nic-rp-0': {"CUSTOM_TOPO_SOCKET0"},
            'nic-rp-1': {"CUSTOM_TOPO_SOCKET1"},
        }
        rps = {
            'accel-rp-A': {"parent_provider_uuid": "numa-anchor-0",
                           "name": "host1_devA"},
            'numa-anchor-0': {
                "parent_provider_uuid": "socket-anchor-0",
                "name": "host1_numa_0",
            },
            'numa-anchor-1': {
                "parent_provider_uuid": "socket-anchor-1",
                "name": "host1_numa_1",
            },
            'socket-anchor-0': {
                "parent_provider_uuid": "host-root",
                "name": "host1_socket_0",
            },
            'socket-anchor-1': {
                "parent_provider_uuid": "host-root",
                "name": "host1_socket_1",
            },
            'host-root': {"parent_provider_uuid": None,
                          "name": "host1"},
            'nic-rp-0': {"parent_provider_uuid": None,
                         "name": "host1:hpc:0000:31:00.0"},
            'nic-rp-1': {"parent_provider_uuid": None,
                         "name": "host1:hpc:0000:b1:00.0"},
        }
        self._install_report(traits, rps)

    def test_mixed_candidates_pass_with_prune(self):
        spec = _spec("socket")
        cand_pass = _candidate(
            mappings={
                "device_profile_0": ["accel-rp-A"],
                "11111111-1111-1111-1111-111111111111": ["nic-rp-0"],
            },
        )
        cand_prune = _candidate(
            mappings={
                "device_profile_0": ["accel-rp-A"],
                "22222222-2222-2222-2222-222222222222": ["nic-rp-1"],
            },
        )
        host = self._host([cand_pass, cand_prune])
        self.assertTrue(self.filt.host_passes(host, spec))
        # Only the matching candidate survived.
        self.assertEqual(1, len(host.allocation_candidates))
        self.assertIn("nic-rp-0",
                      host.allocation_candidates[0]['mappings'][
                          "11111111-1111-1111-1111-111111111111"])

    def test_all_pruned_returns_false(self):
        spec = _spec("socket")
        # Both candidates are cross-socket.
        cand1 = _candidate(
            mappings={
                "device_profile_0": ["accel-rp-A"],
                "11111111-1111-1111-1111-111111111111": ["nic-rp-1"],
            },
        )
        cand2 = _candidate(
            mappings={
                "device_profile_0": ["accel-rp-A"],
                "22222222-2222-2222-2222-222222222222": ["nic-rp-1"],
            },
        )
        host = self._host([cand1, cand2])
        self.assertFalse(self.filt.host_passes(host, spec))


class TestMissingData(_FilterTestBase):
    """Permissive behavior on missing topology data."""

    def test_missing_socket_trait_on_nic_passes(self):
        """NIC has no CUSTOM_TOPO_SOCKET* trait -> pass with WARNING."""
        spec = _spec("socket")
        candidate = _candidate(
            mappings={
                "device_profile_0": ["accel-rp"],
                "11111111-1111-1111-1111-111111111111": ["nic-rp"],
            },
        )
        host = self._host([candidate])
        self._install_report(
            traits={
                "accel-rp": set(),
                "socket-anchor": {"CUSTOM_SOCKET_ROOT"},
                "host-root": set(),
                "nic-rp": set(),  # no topology trait!
            },
            rps={
                "accel-rp": {"parent_provider_uuid": "socket-anchor",
                             "name": "host1_devA"},
                "socket-anchor": {
                    "parent_provider_uuid": "host-root",
                    "name": "host1_socket_0",
                },
                "host-root": {"parent_provider_uuid": None,
                              "name": "host1"},
                "nic-rp": {"parent_provider_uuid": None,
                           "name": "host1:hpc:0000:31:00.0"},
            },
        )
        # Don't fail-closed.
        self.assertTrue(self.filt.host_passes(host, spec))

    def test_missing_socket_anchor_passes(self):
        """Accel RP not under a CUSTOM_SOCKET_ROOT -> pass."""
        spec = _spec("socket")
        candidate = _candidate(
            mappings={
                "device_profile_0": ["accel-rp"],
                "11111111-1111-1111-1111-111111111111": ["nic-rp"],
            },
        )
        host = self._host([candidate])
        self._install_report(
            traits={
                "accel-rp": set(),
                "host-root": set(),
                "nic-rp": {"CUSTOM_TOPO_SOCKET0"},
            },
            rps={
                # Accel parents directly under host root - no anchor.
                "accel-rp": {"parent_provider_uuid": "host-root",
                             "name": "host1_devA"},
                "host-root": {"parent_provider_uuid": None,
                              "name": "host1"},
                "nic-rp": {"parent_provider_uuid": None,
                           "name": "host1:hpc:0000:31:00.0"},
            },
        )
        self.assertTrue(self.filt.host_passes(host, spec))


class TestMultipleResources(_FilterTestBase):
    """Multi-GPU and multi-NIC instances must share the same anchor."""

    def test_multi_accel_must_share_anchor(self):
        spec = _spec("socket")
        candidate = _candidate(
            mappings={
                "device_profile_0": ["accel-rp-0"],
                "device_profile_1": ["accel-rp-1"],  # different socket!
                "11111111-1111-1111-1111-111111111111": ["nic-rp"],
            },
        )
        host = self._host([candidate])
        self._install_report(
            traits={
                "accel-rp-0": set(),
                "accel-rp-1": set(),
                "socket-anchor-0": {"CUSTOM_SOCKET_ROOT"},
                "socket-anchor-1": {"CUSTOM_SOCKET_ROOT"},
                "host-root": set(),
                "nic-rp": {"CUSTOM_TOPO_SOCKET0"},
            },
            rps={
                "accel-rp-0": {
                    "parent_provider_uuid": "socket-anchor-0",
                    "name": "host1_devA",
                },
                "accel-rp-1": {
                    "parent_provider_uuid": "socket-anchor-1",
                    "name": "host1_devB",
                },
                "socket-anchor-0": {
                    "parent_provider_uuid": "host-root",
                    "name": "host1_socket_0",
                },
                "socket-anchor-1": {
                    "parent_provider_uuid": "host-root",
                    "name": "host1_socket_1",
                },
                "host-root": {"parent_provider_uuid": None,
                              "name": "host1"},
                "nic-rp": {"parent_provider_uuid": None,
                           "name": "host1:hpc:0000:31:00.0"},
            },
        )
        self.assertFalse(self.filt.host_passes(host, spec))

    def test_multi_nic_must_share_socket_trait(self):
        spec = _spec("socket")
        candidate = _candidate(
            mappings={
                "device_profile_0": ["accel-rp"],
                "11111111-1111-1111-1111-111111111111": ["nic-rp-A"],
                "22222222-2222-2222-2222-222222222222": ["nic-rp-B"],
            },
        )
        host = self._host([candidate])
        self._install_report(
            traits={
                "accel-rp": set(),
                "socket-anchor": {"CUSTOM_SOCKET_ROOT"},
                "host-root": set(),
                "nic-rp-A": {"CUSTOM_TOPO_SOCKET0"},
                "nic-rp-B": {"CUSTOM_TOPO_SOCKET1"},
            },
            rps={
                "accel-rp": {"parent_provider_uuid": "socket-anchor",
                             "name": "host1_devA"},
                "socket-anchor": {
                    "parent_provider_uuid": "host-root",
                    "name": "host1_socket_0",
                },
                "host-root": {"parent_provider_uuid": None,
                              "name": "host1"},
                "nic-rp-A": {"parent_provider_uuid": None,
                             "name": "host1:hpc:0000:31:00.0"},
                "nic-rp-B": {"parent_provider_uuid": None,
                             "name": "host1:hpc:0000:b1:00.0"},
            },
        )
        self.assertFalse(self.filt.host_passes(host, spec))


class TestNoGpuOrNicInCandidate(_FilterTestBase):
    """When there's nothing to enforce, candidate passes through."""

    def test_only_accel_no_nic_passes(self):
        spec = _spec("socket")
        candidate = _candidate(
            mappings={
                "device_profile_0": ["accel-rp"],
            },
        )
        host = self._host([candidate])
        self._install_report(
            traits={"accel-rp": set()},
            rps={"accel-rp": {"parent_provider_uuid": None,
                              "name": "host1_devA"}},
        )
        self.assertTrue(self.filt.host_passes(host, spec))

    def test_only_nic_no_accel_passes(self):
        spec = _spec("socket")
        candidate = _candidate(
            mappings={
                "11111111-1111-1111-1111-111111111111": ["nic-rp"],
            },
        )
        host = self._host([candidate])
        self._install_report(
            traits={"nic-rp": {"CUSTOM_TOPO_SOCKET0"}},
            rps={"nic-rp": {"parent_provider_uuid": None,
                            "name": "host1:hpc:0000:31:00.0"}},
        )
        self.assertTrue(self.filt.host_passes(host, spec))


class TestHelpers(test.NoDBTestCase):
    """Cover small predicate functions."""

    def test_looks_like_uuid_positive(self):
        self.assertTrue(
            taf._looks_like_uuid("11111111-1111-1111-1111-111111111111"),
        )

    def test_looks_like_uuid_too_short(self):
        self.assertFalse(taf._looks_like_uuid("short"))

    def test_looks_like_uuid_non_string(self):
        self.assertFalse(taf._looks_like_uuid(None))
        self.assertFalse(taf._looks_like_uuid(42))

    def test_looks_like_uuid_wrong_hyphens(self):
        self.assertFalse(
            taf._looks_like_uuid("11111111X1111X1111X1111X111111111111"),
        )


class TestInvalidExtraSpecValue(_FilterTestBase):
    """An invalid hw:cyborg_locality value degrades to 'none'."""

    def test_invalid_value_treated_as_none(self):
        spec = _spec("garbage")
        host = self._host([_candidate({})])
        self.filt._report = mock.MagicMock()
        self.assertTrue(self.filt.host_passes(host, spec))
        self.filt._report.get_provider_traits.assert_not_called()

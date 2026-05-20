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


"""TopologyAffinityFilter

Prunes allocation candidates whose accelerator and NIC resources
don't share the same topology anchor (socket or NUMA node).

Activated by flavor extra-spec::

  hw:cyborg_locality=socket   # GPU and NIC on the same socket
  hw:cyborg_locality=numa     # GPU and NIC on the same NUMA node
  hw:cyborg_locality=none     # no constraint (default)

Requires:

* Cyborg with ``[placement] numa_aware_subtree = True`` (writes the
  socket+NUMA anchor tree -- see the companion Cyborg
  ``feature/amd-v620`` fork's
  ``cyborg.conductor.manager._get_or_create_socket_subprovider``).
* Cyborg with the ``nic_topology_driver`` enabled (PATCHes
  ``CUSTOM_TOPO_SOCKET<n>`` / ``CUSTOM_TOPO_NUMA<n>`` traits onto
  Neutron's flat NIC RPs).
* This filter included in ``[filter_scheduler] enabled_filters``.
  It is NOT in the default list -- operators opt in.

Behavior on missing data: when topology info is unavailable for a
given RP (no socket anchor, no CUSTOM_TOPO_SOCKET trait, Placement
timeout), the filter logs a warning and treats the candidate as
*passing*. Failing closed would block all scheduling during transient
Placement outages or partial Cyborg deployment; we prefer the
permissive failure mode because operators can fix the topology
metadata in-place via Cyborg, but cannot recover scheduling time
once it's lost.

DROP WHEN NOVA LANDS nova-spec-numa-topology-with-rps.
"""

from oslo_log import log as logging

from nova import exception
from nova.scheduler.client import report as report_client
from nova.scheduler import filters


LOG = logging.getLogger(__name__)


# Flavor extra-spec key.
_EXTRA_SPEC_KEY = "hw:cyborg_locality"

# Recognized values.
_VAL_SOCKET = "socket"
_VAL_NUMA = "numa"
_VAL_NONE = "none"
_VALID_VALUES = frozenset([_VAL_SOCKET, _VAL_NUMA, _VAL_NONE])

# Anchor traits emitted by Cyborg's socket+NUMA layer (see
# cyborg/conductor/manager.py).
_SOCKET_ANCHOR_TRAIT = "CUSTOM_SOCKET_ROOT"
_NUMA_ANCHOR_TRAIT = "HW_NUMA_ROOT"

# Trait prefixes PATCHed onto Neutron NIC RPs by Cyborg's
# nic_topology_driver. Must match the driver's defaults.
_NIC_SOCKET_PREFIX = "CUSTOM_TOPO_SOCKET"
_NIC_NUMA_PREFIX = "CUSTOM_TOPO_NUMA"

# Heuristic accel-suffix prefix (Cyborg-style; see
# nova.accelerator.cyborg.get_device_profile_group_requester_id).
_ACCEL_SUFFIX_PREFIX = "device_profile_"

# NIC-side identification by traits on the mapped RP (vnic-type-
# agnostic - we accept any of these as "NIC RP").
_NIC_RP_TRAITS = frozenset([
    "CUSTOM_VNIC_TYPE_DIRECT",
    "CUSTOM_VNIC_TYPE_VDPA",
    "CUSTOM_VNIC_TYPE_DIRECT_PHYSICAL",
    "CUSTOM_VNIC_TYPE_VIRTIO_FORWARDER",
])
_NIC_RP_TRAIT_PREFIXES = ("CUSTOM_PHYSNET_",)

_UUID_LEN = 36


def _looks_like_uuid(s):
    if not isinstance(s, str) or len(s) != _UUID_LEN:
        return False
    return s[8] == '-' and s[13] == '-' and s[18] == '-' and s[23] == '-'


class TopologyAffinityFilter(
    filters.BaseHostFilter, filters.CandidateFilterMixin,
):
    """Socket/NUMA cross-product affinity filter.

    See module docstring.
    """

    # Locality is request-level; the candidate set within a host
    # changes per instance, so we cannot reuse a prior decision.
    run_filter_once_per_request = False
    RUN_ON_REBUILD = False

    def __init__(self):
        super().__init__()
        self._report = None
        # Per-host-pass caches. Cleared at the start of each
        # ``host_passes`` call to bound memory; the same RP UUID may
        # legitimately appear in many candidates within a single
        # host's allocation_candidates list, so caching across
        # candidates is the main saving.
        #
        # rp_uuid -> str_anchor_id  (e.g. "socket:0" or "numa:0")
        self._anchor_cache_socket = {}
        self._anchor_cache_numa = {}
        # rp_uuid -> set(trait_name)
        self._trait_cache = {}
        # rp_uuid -> RP dict (or None on 404)
        self._rp_cache = {}

    # ---- public scheduler-facing entry point ----

    def host_passes(self, host_state, spec_obj):
        try:
            extra_specs = spec_obj.flavor.extra_specs or {}
        except AttributeError:
            return True
        locality = extra_specs.get(_EXTRA_SPEC_KEY, _VAL_NONE)
        if locality not in _VALID_VALUES:
            LOG.warning(
                'TopologyAffinityFilter: invalid %s=%r; treating as '
                '"none" for host %s.',
                _EXTRA_SPEC_KEY, locality, host_state.host,
            )
            locality = _VAL_NONE
        if locality == _VAL_NONE:
            return True

        # Fresh per-call caches (a scheduling cycle re-instantiates
        # the filter; even if it didn't, candidate-shared lookups
        # within one host are the main saving).
        self._anchor_cache_socket.clear()
        self._anchor_cache_numa.clear()
        self._trait_cache.clear()
        self._rp_cache.clear()

        # If the host has no allocation_candidates attached (rebuild
        # path or an out-of-band invocation), let it pass.
        candidates = getattr(host_state, 'allocation_candidates', None)
        if not candidates:
            return True

        report = self._get_report_client()

        # Use the candidate filter mixin to prune and return.
        def _filter_func(candidate):
            return self._candidate_satisfies(
                candidate, locality, host_state, report,
            )

        return bool(self.filter_candidates(host_state, _filter_func))

    # ---- internals ----

    def _get_report_client(self):
        """Lazily instantiate a SchedulerReportClient.

        The standalone client carries its own cache; we let it manage
        keystone auth and Placement microversion negotiation.
        """
        if self._report is None:
            self._report = report_client.SchedulerReportClient()
        return self._report

    def _candidate_satisfies(self, candidate, locality, host_state, report):
        """Return True if ``candidate`` satisfies the topology constraint.

        On *any* exception or missing-data condition, returns True
        with a logged warning. See the module docstring for why we
        prefer the permissive failure mode.
        """
        try:
            accel_rps, nic_rps = self._classify_candidate_rps(
                candidate, report,
            )
        except Exception as e:
            LOG.warning(
                'TopologyAffinityFilter: candidate classification '
                'failed on host %s: %s. Passing candidate.',
                host_state.host, e,
            )
            return True

        if not accel_rps or not nic_rps:
            # Nothing to enforce - no GPU+NIC pair in this candidate.
            # Pass through (the filter only constrains hybrid bookings).
            return True

        if locality == _VAL_SOCKET:
            cache = self._anchor_cache_socket
            resolver = self._resolve_accel_socket_anchor
            nic_resolver = self._resolve_nic_socket_id
        else:  # numa
            cache = self._anchor_cache_numa
            resolver = self._resolve_accel_numa_anchor
            nic_resolver = self._resolve_nic_numa_id

        # Collect every accel anchor and every NIC topology identifier.
        accel_anchors = set()
        for rp_uuid in accel_rps:
            anchor = resolver(rp_uuid, report, cache)
            if anchor is None:
                LOG.warning(
                    'TopologyAffinityFilter: no %s anchor for accel '
                    'RP %s on host %s; passing candidate.',
                    locality, rp_uuid, host_state.host,
                )
                return True
            accel_anchors.add(anchor)

        nic_anchors = set()
        for rp_uuid in nic_rps:
            anchor = nic_resolver(rp_uuid, report)
            if anchor is None:
                LOG.warning(
                    'TopologyAffinityFilter: no %s topology trait on '
                    'NIC RP %s (host %s); passing candidate. Confirm '
                    'the nic_topology_driver is enabled on this '
                    'compute.',
                    locality, rp_uuid, host_state.host,
                )
                return True
            nic_anchors.add(anchor)

        # All accel + NIC RPs must collapse to a single anchor id.
        all_anchors = accel_anchors | nic_anchors
        if len(all_anchors) == 1:
            return True
        LOG.debug(
            'TopologyAffinityFilter: prune candidate (host %s) - '
            'accel anchors %r != NIC anchors %r.',
            host_state.host, accel_anchors, nic_anchors,
        )
        return False

    def _classify_candidate_rps(self, candidate, report):
        """Walk a candidate's mappings, return (accel_rps, nic_rps).

        Identification is two-pronged so we don't depend on any one
        signal:

        * Accel: the requester_id (mapping key) starts with
          ``device_profile_`` (Cyborg's convention).
        * NIC: the requester_id is UUID-shaped (Neutron port suffix),
          OR the mapped RP carries a vnic-type / physnet trait.

        The trait-based fallback for NIC covers the case where a
        request group was synthesized server-side without a port UUID
        as its requester id.
        """
        mappings = candidate.get('mappings') or {}
        accel_rps = set()
        nic_rps = set()
        for requester_id, rp_uuids in mappings.items():
            if not rp_uuids:
                continue
            uniq = set(rp_uuids)
            if requester_id.startswith(_ACCEL_SUFFIX_PREFIX):
                accel_rps.update(uniq)
                continue
            if _looks_like_uuid(requester_id):
                nic_rps.update(uniq)
                continue
            # Fall back to trait inspection for ambiguous suffixes.
            for rp_uuid in uniq:
                traits = self._get_rp_traits(rp_uuid, report)
                if self._traits_indicate_nic(traits):
                    nic_rps.add(rp_uuid)
        return accel_rps, nic_rps

    def _traits_indicate_nic(self, traits):
        if not traits:
            return False
        if traits & _NIC_RP_TRAITS:
            return True
        for t in traits:
            for p in _NIC_RP_TRAIT_PREFIXES:
                if t.startswith(p):
                    return True
        return False

    def _resolve_accel_socket_anchor(self, rp_uuid, report, cache):
        """Walk the accel RP's ancestry to a CUSTOM_SOCKET_ROOT anchor.

        Returns "socket:<uuid>" identifier (UUID of the anchor RP) or
        None if not found. Walks at most a few hops upward
        (device -> NUMA anchor -> socket anchor -> host root).
        """
        if rp_uuid in cache:
            return cache[rp_uuid]
        result = self._walk_to_anchor(
            rp_uuid, report, _SOCKET_ANCHOR_TRAIT,
        )
        cache[rp_uuid] = result
        return result

    def _resolve_accel_numa_anchor(self, rp_uuid, report, cache):
        """Walk to a HW_NUMA_ROOT anchor."""
        if rp_uuid in cache:
            return cache[rp_uuid]
        result = self._walk_to_anchor(
            rp_uuid, report, _NUMA_ANCHOR_TRAIT,
        )
        cache[rp_uuid] = result
        return result

    def _walk_to_anchor(self, rp_uuid, report, anchor_trait, max_hops=8):
        """Walk up the RP tree looking for ``anchor_trait``.

        :returns: ``"<anchor_trait>:<rp_uuid>"`` for the matched anchor,
                  or None if we hit the root or hop limit first.
        """
        seen = set()
        cur = rp_uuid
        for _ in range(max_hops):
            if cur in seen:
                return None
            seen.add(cur)
            traits = self._get_rp_traits(cur, report)
            if anchor_trait in traits:
                return "%s:%s" % (anchor_trait, cur)
            rp = self._get_rp(cur, report)
            if not rp:
                return None
            parent = rp.get('parent_provider_uuid')
            if not parent:
                return None
            cur = parent
        return None

    def _resolve_nic_socket_id(self, rp_uuid, report):
        """Extract the socket integer from CUSTOM_TOPO_SOCKET<n>."""
        return self._read_nic_topology_id(
            rp_uuid, report, _NIC_SOCKET_PREFIX, _SOCKET_ANCHOR_TRAIT,
        )

    def _resolve_nic_numa_id(self, rp_uuid, report):
        return self._read_nic_topology_id(
            rp_uuid, report, _NIC_NUMA_PREFIX, _NUMA_ANCHOR_TRAIT,
        )

    def _read_nic_topology_id(self, rp_uuid, report, prefix, anchor_kind):
        """Find a single CUSTOM_TOPO_*<n> trait on the NIC RP.

        Returns ``"<anchor_kind>:<n>"`` so it's comparable to the
        accel-side ``_walk_to_anchor`` result via the same id space.
        We use the SAME prefix (CUSTOM_SOCKET_ROOT / HW_NUMA_ROOT)
        for accel anchors, but only the *integer* matters for NIC RPs
        - we rebuild the comparable id by tagging it with the anchor
        kind.

        Note: accel returns a UUID-tagged id (because the anchor's
        identity is its UUID), while NIC returns an integer-tagged
        id. To make them comparable, we normalize: both sides return
        ``"<anchor_kind>:<integer>"`` for socket/NUMA layers. See
        ``_walk_to_anchor_integer`` for the integer extraction on
        the accel side.
        """
        traits = self._get_rp_traits(rp_uuid, report)
        matching = [t for t in traits if t.startswith(prefix)]
        if not matching:
            return None
        if len(matching) > 1:
            LOG.warning(
                'TopologyAffinityFilter: NIC RP %s has multiple %s '
                'traits %r; picking the first sorted.',
                rp_uuid, prefix, matching,
            )
        # CUSTOM_TOPO_SOCKET0 -> 0
        chosen = sorted(matching)[0]
        suffix = chosen[len(prefix):]
        try:
            n = int(suffix)
        except ValueError:
            LOG.warning(
                'TopologyAffinityFilter: cannot parse integer from '
                'trait %r on NIC RP %s.', chosen, rp_uuid,
            )
            return None
        return "%s:int:%d" % (anchor_kind, n)

    # We need accel-side resolution to ALSO return an integer-tagged
    # id, so both sides land in the same id space. Replace the
    # _walk_to_anchor return for the comparison-time helpers:
    def _resolve_accel_socket_anchor_int(self, rp_uuid, report, cache):
        """Like _resolve_accel_socket_anchor but returns integer id.

        Reads the anchor's name (``<host>_socket_<n>``) and extracts
        ``n`` so it matches the NIC-side integer-tagged form.
        """
        return self._resolve_accel_anchor_int(
            rp_uuid, report, cache,
            anchor_trait=_SOCKET_ANCHOR_TRAIT,
            name_marker='_socket_',
        )

    def _resolve_accel_numa_anchor_int(self, rp_uuid, report, cache):
        return self._resolve_accel_anchor_int(
            rp_uuid, report, cache,
            anchor_trait=_NUMA_ANCHOR_TRAIT,
            name_marker='_numa_',
        )

    def _resolve_accel_anchor_int(
        self, rp_uuid, report, cache, anchor_trait, name_marker,
    ):
        cached = cache.get(rp_uuid)
        if cached is not None:
            return cached if cached != '__miss__' else None
        anchor_id = self._walk_to_anchor(rp_uuid, report, anchor_trait)
        if anchor_id is None:
            cache[rp_uuid] = '__miss__'
            return None
        # anchor_id is "<anchor_trait>:<rp_uuid>"; we need to fetch
        # the anchor's name to extract the integer after name_marker.
        anchor_rp_uuid = anchor_id.split(':', 1)[1]
        anchor_rp = self._get_rp(anchor_rp_uuid, report)
        if not anchor_rp:
            cache[rp_uuid] = '__miss__'
            return None
        name = anchor_rp.get('name') or ''
        idx = name.find(name_marker)
        if idx < 0:
            cache[rp_uuid] = '__miss__'
            return None
        tail = name[idx + len(name_marker):]
        try:
            n = int(tail)
        except ValueError:
            cache[rp_uuid] = '__miss__'
            return None
        normalized = "%s:int:%d" % (anchor_trait, n)
        cache[rp_uuid] = normalized
        return normalized

    # ---- Placement plumbing (cached) ----

    def _get_rp_traits(self, rp_uuid, report):
        if rp_uuid in self._trait_cache:
            return self._trait_cache[rp_uuid]
        try:
            info = report.get_provider_traits(_FakeCtx(), rp_uuid)
            traits = set(info.traits)
        except exception.ResourceProviderTraitRetrievalFailed:
            traits = set()
        except Exception as e:
            LOG.debug(
                'TopologyAffinityFilter: get_provider_traits(%s) '
                'failed: %s', rp_uuid, e,
            )
            traits = set()
        self._trait_cache[rp_uuid] = traits
        return traits

    def _get_rp(self, rp_uuid, report):
        if rp_uuid in self._rp_cache:
            return self._rp_cache[rp_uuid]
        try:
            rp = report._get_resource_provider(_FakeCtx(), rp_uuid)
        except Exception as e:
            LOG.debug(
                'TopologyAffinityFilter: _get_resource_provider(%s) '
                'failed: %s', rp_uuid, e,
            )
            rp = None
        self._rp_cache[rp_uuid] = rp
        return rp


# Wire up the comparable resolvers to the host_passes path.
TopologyAffinityFilter._resolve_accel_socket_anchor = (
    TopologyAffinityFilter._resolve_accel_socket_anchor_int
)
TopologyAffinityFilter._resolve_accel_numa_anchor = (
    TopologyAffinityFilter._resolve_accel_numa_anchor_int
)


class _FakeCtx:
    """Minimal stand-in for the request context.

    SchedulerReportClient's read APIs only consult ``global_id`` for
    request-id passthrough; we don't have a real RequestContext in
    filter scope and constructing one with full auth would couple the
    filter to keystone auth state. The placement read calls
    themselves are authed via the SchedulerReportClient's own
    keystone session, not via this context object.
    """
    global_id = None

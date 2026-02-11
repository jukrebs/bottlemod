"""
Tests for BottleMod-SH (Storage Hierarchy Extension)

Tests the core functionality:
1. LogicalAccessProfile construction and derivatives
2. StorageTier creation with different configurations
3. TierMapping with various cache behavior patterns
4. Cache behavior models (DirectHitRate, StackDistance, PhaseBased)
5. Resource derivation from (A, Q, H) to standard BottleMod R_{R,l}(p)
6. Integration with TaskExecution
"""

import unittest
from typing import cast

from bottlemod.func import Func
from bottlemod.ppoly import PPoly
from bottlemod.storage_hierarchy import (
    AccessType,
    DirectHitRateModel,
    LogicalAccessProfile,
    PhaseBasedCacheModel,
    ResourceType,
    StackDistanceModel,
    StorageHierarchyTask,
    StorageTier,
    TierMapping,
    derive_tier_resources,
)


class TestLogicalAccessProfile(unittest.TestCase):
    def test_sequential_read_creation(self):
        profile = LogicalAccessProfile.sequential_read(
            name="test_file",
            total_bytes=1e9,
            max_progress=100.0,
        )

        self.assertEqual(profile.name, "test_file")
        self.assertIsNotNone(profile.A_read)
        self.assertIsNotNone(profile.Q_read)
        assert profile.A_read is not None
        assert profile.Q_read is not None

        self.assertAlmostEqual(profile.A_read(100.0), 1e9, delta=1e3)

        self.assertTrue(
            cast(float, profile.Q_read(100.0)) < cast(float, profile.A_read(100.0))
        )

    def test_random_read_creation(self):
        profile = LogicalAccessProfile.random_read(
            name="random_data",
            total_bytes=1e6,
            max_progress=100.0,
            request_size=4096,
        )
        assert profile.Q_read is not None

        expected_ops = 1e6 / 4096
        self.assertAlmostEqual(profile.Q_read(100.0), expected_ops, delta=1)

    def test_derivatives(self):
        profile = LogicalAccessProfile.sequential_read(
            name="test",
            total_bytes=1000.0,
            max_progress=100.0,
        )

        A_prime = profile.get_derivative(AccessType.READ)
        self.assertAlmostEqual(A_prime(50.0), 10.0, delta=0.01)

    def test_defaults_for_writes(self):
        profile = LogicalAccessProfile(
            name="read_only",
            A_read=PPoly([0, 100], [[10]]),
        )

        self.assertIsNotNone(profile.A_write)
        self.assertIsNotNone(profile.Q_write)
        assert profile.A_write is not None
        assert profile.Q_write is not None
        self.assertEqual(profile.A_write(50), 0)


class TestStorageTier(unittest.TestCase):
    def test_memory_tier(self):
        tier = StorageTier.memory(
            name="DRAM",
            bandwidth_GBps=25.0,
            capacity_GB=16.0,
        )

        self.assertEqual(tier.name, "DRAM")
        self.assertEqual(tier.tier_index, 0)
        assert tier.capacity is not None
        self.assertAlmostEqual(tier.capacity, 16e9, delta=1e6)

        bw = tier.get_resource_input(ResourceType.BANDWIDTH, AccessType.READ)
        assert bw is not None
        self.assertAlmostEqual(bw(0), 25e9, delta=1e6)

    def test_nvme_tier(self):
        tier = StorageTier.nvme_ssd(
            name="NVMe",
            bandwidth_GBps=3.0,
            iops_k=500.0,
        )

        self.assertEqual(tier.tier_index, 1)

        iops = tier.get_resource_input(ResourceType.IOPS, AccessType.READ)
        assert iops is not None
        self.assertAlmostEqual(iops(0), 500e3, delta=100)

    def test_hdd_tier(self):
        tier = StorageTier.hdd(
            name="HDD",
            bandwidth_MBps=150.0,
            iops=150.0,
        )

        self.assertEqual(tier.tier_index, 3)

        bw = tier.get_resource_input(ResourceType.BANDWIDTH, AccessType.READ)
        assert bw is not None
        self.assertAlmostEqual(bw(0), 150e6, delta=1e3)


class TestTierMapping(unittest.TestCase):
    def test_all_from_tier(self):
        mapping = TierMapping.all_from_tier(
            dataset_name="data",
            tier_index=1,
            progress_range=(0, 100),
        )

        H = mapping.get_hit_fraction(1, AccessType.READ)
        self.assertAlmostEqual(H(50), 1.0, delta=0.01)

        H_other = mapping.get_hit_fraction(0, AccessType.READ)
        self.assertAlmostEqual(H_other(50), 0.0, delta=0.01)

    def test_cold_then_warm(self):
        mapping = TierMapping.cold_then_warm(
            dataset_name="data",
            cold_tier=1,
            warm_tier=0,
            warmup_progress=50.0,
            progress_range=(0, 100),
            warm_hit_rate=1.0,
        )

        H_warm = mapping.get_hit_fraction(0, AccessType.READ)
        H_cold = mapping.get_hit_fraction(1, AccessType.READ)

        self.assertAlmostEqual(H_warm(25), 0.0, delta=0.01)
        self.assertAlmostEqual(H_cold(25), 1.0, delta=0.01)

        self.assertAlmostEqual(H_warm(75), 1.0, delta=0.01)
        self.assertAlmostEqual(H_cold(75), 0.0, delta=0.01)

    def test_constant_hit_rate(self):
        mapping = TierMapping.constant_hit_rate(
            dataset_name="data",
            cache_tier=0,
            backing_tier=1,
            hit_rate=0.8,
            progress_range=(0, 100),
        )

        H_cache = mapping.get_hit_fraction(0, AccessType.READ)
        H_backing = mapping.get_hit_fraction(1, AccessType.READ)

        self.assertAlmostEqual(H_cache(50), 0.8, delta=0.01)
        self.assertAlmostEqual(H_backing(50), 0.2, delta=0.01)

    def test_validation(self):
        mapping = TierMapping.constant_hit_rate(
            dataset_name="data",
            cache_tier=0,
            backing_tier=1,
            hit_rate=0.5,
            progress_range=(0, 100),
        )

        mapping.validate()


class TestCacheBehaviorModels(unittest.TestCase):
    def test_direct_hit_rate_model(self):
        model = DirectHitRateModel(
            hit_rate_func=lambda p: 0.9 if p > 50 else 0.1,
            cache_tier=0,
        )

        profile = LogicalAccessProfile.sequential_read("test", 1000, 100)
        tiers = [
            StorageTier.memory(),
            StorageTier.sata_ssd(),
        ]

        mapping = model.compute_tier_mapping(profile, tiers, (0, 100))

        H_cache_low = mapping.get_hit_fraction(0, AccessType.READ)
        self.assertTrue(cast(float, H_cache_low(25)) < 0.5)
        self.assertTrue(cast(float, H_cache_low(75)) > 0.5)

    def test_stack_distance_uniform(self):
        model = StackDistanceModel.uniform_reuse(working_set_size=1e9)

        profile = LogicalAccessProfile.random_read("test", 1e6, 100, 4096)
        tiers = [
            StorageTier.memory(capacity_GB=16),
            StorageTier.sata_ssd(),
        ]

        mapping = model.compute_tier_mapping(profile, tiers, (0, 100))

        H_mem = mapping.get_hit_fraction(0, AccessType.READ)
        expected_hit = min(1.0, 16e9 / 1e9)
        self.assertAlmostEqual(H_mem(50), expected_hit, delta=0.1)

    def test_stack_distance_no_reuse(self):
        model = StackDistanceModel.streaming_no_reuse()

        profile = LogicalAccessProfile.sequential_read("test", 1e9, 100)
        tiers = [
            StorageTier.memory(capacity_GB=16),
            StorageTier.sata_ssd(),
        ]

        mapping = model.compute_tier_mapping(profile, tiers, (0, 100))

        H_mem = mapping.get_hit_fraction(0, AccessType.READ)
        self.assertAlmostEqual(H_mem(50), 0.0, delta=0.01)

    def test_phase_based_model(self):
        phases = [
            PhaseBasedCacheModel.Phase(0, 50, 0.0),
            PhaseBasedCacheModel.Phase(50, 100, 0.9),
        ]
        model = PhaseBasedCacheModel(phases, cache_tier=0)

        profile = LogicalAccessProfile.sequential_read("test", 1e9, 100)
        tiers = [
            StorageTier.memory(),
            StorageTier.sata_ssd(),
        ]

        mapping = model.compute_tier_mapping(profile, tiers, (0, 100))

        H_cache = mapping.get_hit_fraction(0, AccessType.READ)
        self.assertAlmostEqual(H_cache(25), 0.0, delta=0.01)
        self.assertAlmostEqual(H_cache(75), 0.9, delta=0.01)


class TestResourceDerivation(unittest.TestCase):
    def test_derive_tier_resources(self):
        profile = LogicalAccessProfile.sequential_read("file", 1e9, 100)

        tiers = [
            StorageTier.memory(time_range=(0, 100)),
            StorageTier.sata_ssd(time_range=(0, 100)),
        ]

        mapping = TierMapping.all_from_tier("file", 1, (0, 100))

        reqs, inputs = derive_tier_resources(profile, mapping, tiers)

        self.assertEqual(len(reqs), 8)
        self.assertEqual(len(inputs), 8)

        for r in reqs:
            self.assertIsInstance(r, PPoly)
        for i in inputs:
            self.assertIsInstance(i, PPoly)

    def test_derived_resources_are_rates(self):
        profile = LogicalAccessProfile.sequential_read("file", 1000, 100)

        tiers = [StorageTier.memory(time_range=(0, 100))]
        mapping = TierMapping.all_from_tier("file", 0, (0, 100))

        reqs, _ = derive_tier_resources(profile, mapping, tiers)

        R_bw = reqs[0]
        self.assertAlmostEqual(R_bw(50), 10.0, delta=0.01)
        self.assertAlmostEqual(R_bw(0), R_bw(100), delta=0.01)


class TestStorageHierarchyTask(unittest.TestCase):
    def test_task_creation(self):
        profile = LogicalAccessProfile.sequential_read("data", 1e6, 100)
        tiers = [
            StorageTier.memory(time_range=(0, 100)),
            StorageTier.sata_ssd(time_range=(0, 100)),
        ]
        mapping = TierMapping.all_from_tier("data", 1, (0, 100))

        cpu_func = PPoly([0, 100], [[1]])
        data_func = Func([0, 1e6], [[100 / 1e6, 0]])

        sh_task = StorageHierarchyTask(
            access_profiles=[profile],
            tier_mappings=[mapping],
            tiers=tiers,
            cpu_funcs=[cpu_func],
            data_funcs=[data_func],
        )

        task = sh_task.to_task()

        self.assertEqual(len(task.cpu_funcs), 9)
        self.assertEqual(len(task.data_funcs), 1)

    def test_resource_labels(self):
        profile = LogicalAccessProfile.sequential_read("myfile", 1e6, 100)
        tiers = [StorageTier.memory(name="RAM")]
        mapping = TierMapping.all_from_tier("myfile", 0, (0, 100))

        sh_task = StorageHierarchyTask(
            access_profiles=[profile],
            tier_mappings=[mapping],
            tiers=tiers,
            cpu_funcs=[PPoly([0, 100], [[1]])],
            data_funcs=[Func([0, 1e6], [[100 / 1e6, 0]])],
        )

        labels = sh_task.get_resource_labels()

        self.assertIn("CPU_0", labels)
        self.assertTrue(any("RAM" in l for l in labels))
        self.assertTrue(any("bw_read" in l for l in labels))


if __name__ == "__main__":
    unittest.main()

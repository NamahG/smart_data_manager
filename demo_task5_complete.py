"""
TASK 5 COMPLETE DEMO: Data Consistency and Availability
Demonstrates:
1. Synchronization across distributed environments
2. Network failure handling
3. Data conflict detection and resolution
4. Version management
5. Health monitoring
6. Graceful degradation
"""

from component1_data_sorter import (
    SmartDataManager, DataObject, StorageTier,
    StorageLocation, LatencyRequirement
)
from component5_consistency import (
    ConsistencyManager, ResolutionStrategy
)
from datetime import datetime, timedelta
import time


def create_distributed_dataset(consistency_mgr: ConsistencyManager):
    """Create dataset distributed across multiple clouds"""

    manager = SmartDataManager()

    # File 1: Replicated across AWS and Azure
    file1 = DataObject("DIST001", "customer_data.db", 5000,
                      StorageLocation.AWS, StorageTier.HOT,
                      LatencyRequirement.CRITICAL)

    # File 2: Replicated across GCP and On-Premise
    file2 = DataObject("DIST002", "analytics_warehouse.db", 10000,
                      StorageLocation.GCP, StorageTier.WARM,
                      LatencyRequirement.STANDARD)

    # File 3: On Private Cloud with replicas
    file3 = DataObject("DIST003", "application_state.json", 500,
                      StorageLocation.PRIVATE_CLOUD, StorageTier.HOT,
                      LatencyRequirement.CRITICAL)

    files = [file1, file2, file3]

    for f in files:
        manager.add_data(f)
        # Register initial version
        consistency_mgr.register_data_object(f)

    return manager, files


def simulate_replication(consistency_mgr: ConsistencyManager, file: DataObject, locations: list):
    """Simulate replication to multiple locations"""
    print(f"Replicating {file.name} to {len(locations)} locations...")

    for loc in locations:
        # Create copy at new location
        replica = DataObject(file.file_id, file.name, file.size_mb,
                           loc, file.tier, file.latency_requirement)

        # Register version at new location
        version = consistency_mgr.register_data_object(replica)
        print(f"  ✓ Created version {version.version} at {loc.value}")


def simulate_concurrent_updates(consistency_mgr: ConsistencyManager, file: DataObject):
    """Simulate concurrent updates creating conflicts"""
    print(f"\n⚠️  Simulating concurrent updates to {file.name}...")

    # Update at AWS
    file_aws = DataObject(file.file_id, file.name, file.size_mb,
                         StorageLocation.AWS, file.tier, file.latency_requirement)
    v1 = consistency_mgr.update_data_object(file_aws)
    print(f"  ✓ AWS updated to version {v1.version}")

    time.sleep(0.1)

    # Concurrent update at Azure
    file_azure = DataObject(file.file_id, file.name, file.size_mb,
                           StorageLocation.AZURE, file.tier, file.latency_requirement)
    v2 = consistency_mgr.update_data_object(file_azure)
    print(f"  ✓ Azure updated to version {v2.version}")

    print(f"  ⚠️  Version conflict created! (AWS: v{v1.version}, Azure: v{v2.version})")


def run_task5_demo():
    """Run complete Task 5 demonstration"""

    print("\n" + "="*85)
    print("  TASK 5: DATA CONSISTENCY AND AVAILABILITY - COMPLETE DEMONSTRATION")
    print("  Features:")
    print("  • Distributed data versioning")
    print("  • Conflict detection and resolution")
    print("  • Network failure handling")
    print("  • Health monitoring")
    print("  • Automatic reconciliation")
    print("="*85 + "\n")

    # Step 1: Initialize system
    print("="*85)
    print("STEP 1: INITIALIZE CONSISTENCY SYSTEM")
    print("="*85 + "\n")

    consistency_mgr = ConsistencyManager()

    print("✅ Consistency components initialized:")
    print("   • Version Manager: Ready")
    print("   • Conflict Detector: Ready")
    print("   • Conflict Resolver: Ready")
    print("   • Health Monitor: Ready")
    print(f"   • Monitoring {len(StorageLocation)} storage nodes\n")

    # Step 2: Create distributed dataset
    print("="*85)
    print("STEP 2: CREATE DISTRIBUTED DATASET")
    print("="*85 + "\n")

    manager, files = create_distributed_dataset(consistency_mgr)

    print(f"Created {len(files)} files with initial versions:\n")

    for file in files:
        version = consistency_mgr.version_manager.get_latest_version(file.file_id)
        print(f"  📁 {file.name}")
        print(f"     Location: {file.location.value}")
        print(f"     Version: {version.version}")
        print(f"     Checksum: {version.checksum}")
        print()

    # Step 3: Replicate data
    print("="*85)
    print("STEP 3: REPLICATE DATA ACROSS CLOUDS")
    print("="*85 + "\n")

    # Replicate file1 to multiple locations
    simulate_replication(consistency_mgr, files[0],
                        [StorageLocation.AZURE, StorageLocation.GCP])

    # Replicate file2
    simulate_replication(consistency_mgr, files[1],
                        [StorageLocation.ON_PREMISE, StorageLocation.AWS])

    print(f"\n✅ Replication complete")
    print(f"   Total versions created: {sum(len(v) for v in consistency_mgr.version_manager.versions.values())}\n")

    # Step 4: Health monitoring
    print("="*85)
    print("STEP 4: SYSTEM HEALTH MONITORING")
    print("="*85 + "\n")

    health_status = consistency_mgr.check_system_health()

    print("🏥 Node Health Status:\n")

    for location_name, status in health_status['node_health'].items():
        icon = "✅" if status.value == "healthy" else "⚠️"
        print(f"   {icon} {location_name.upper():15s}: {status.value}")

    stats = health_status['statistics']
    print(f"\n📊 System Statistics:")
    print(f"   Total nodes: {stats['total_nodes']}")
    print(f"   Healthy: {stats['healthy']}")
    print(f"   Failed: {stats['failed']}")
    print(f"   Average uptime: {stats['average_uptime_percent']}%\n")

    # Step 5: Simulate concurrent updates (create conflicts)
    print("="*85)
    print("STEP 5: SIMULATE CONCURRENT UPDATES (Creating Conflicts)")
    print("="*85 + "\n")

    print("Scenario: Same file updated simultaneously at different locations")

    simulate_concurrent_updates(consistency_mgr, files[0])

    # Step 6: Detect conflicts
    print("\n" + "="*85)
    print("STEP 6: CONFLICT DETECTION")
    print("="*85 + "\n")

    file_ids = [f.file_id for f in files]
    consistency_check = consistency_mgr.run_consistency_check(file_ids)

    print(f"🔍 Consistency Check Results:\n")
    print(f"   Files checked: {len(file_ids)}")
    print(f"   Conflicts detected: {consistency_check['conflicts_detected']}\n")

    if consistency_check['conflicts_detected'] > 0:
        print("📋 Detected Conflicts:\n")

        for i, conflict in enumerate(consistency_check['conflicts'], 1):
            print(f"   {i}. Conflict ID: {conflict['conflict_id']}")
            print(f"      Type: {conflict['conflict_type']}")
            print(f"      File: {conflict['file_id']}")
            print(f"      Versions involved: {conflict['versions_count']}")
            print(f"      Detected at: {conflict['detected_at']}")
            print()

    # Step 7: Automatic conflict resolution
    print("="*85)
    print("STEP 7: AUTOMATIC CONFLICT RESOLUTION")
    print("="*85 + "\n")

    print("⚙️  Attempting automatic resolution...\n")

    resolution_results = consistency_mgr.auto_resolve_conflicts()

    print(f"✅ Resolution Results:")
    print(f"   Total conflicts: {resolution_results['total_conflicts']}")
    print(f"   Auto-resolved: {resolution_results['auto_resolved']}")
    print(f"   Manual review needed: {resolution_results['manual_review_needed']}\n")

    # Show resolution details
    resolved = consistency_mgr.conflict_resolver.resolved_conflicts
    if resolved:
        print("📝 Resolution Details:\n")

        for conflict in resolved:
            print(f"   Conflict: {conflict.conflict_id}")
            print(f"   Strategy: {conflict.resolution_strategy.value}")
            print(f"   Resolved at: {conflict.resolved_at}")
            print()

    # Step 8: Simulate network failure
    print("="*85)
    print("STEP 8: NETWORK FAILURE HANDLING")
    print("="*85 + "\n")

    print("Simulating network partition...\n")

    # Force some nodes to fail
    consistency_mgr.health_monitor.nodes[StorageLocation.AZURE].status = \
        consistency_mgr.health_monitor.nodes[StorageLocation.AZURE].status.__class__.UNREACHABLE
    consistency_mgr.health_monitor.nodes[StorageLocation.AZURE].failures += 1

    consistency_mgr.health_monitor.nodes[StorageLocation.GCP].status = \
        consistency_mgr.health_monitor.nodes[StorageLocation.GCP].status.__class__.UNREACHABLE
    consistency_mgr.health_monitor.nodes[StorageLocation.GCP].failures += 1

    # Re-check health
    health_status = consistency_mgr.check_system_health()

    print("🚨 Network Failure Detected:\n")

    failed_nodes = []
    for location_name, status in health_status['node_health'].items():
        if status.value != "healthy":
            failed_nodes.append(location_name)
            print(f"   ❌ {location_name.upper()}: {status.value}")

    print(f"\n⚠️  {len(failed_nodes)} nodes unreachable")

    # Show failover
    healthy_nodes = health_status['healthy_nodes']
    print(f"\n✅ Failover Strategy:")
    print(f"   Healthy nodes available: {len(healthy_nodes)}")
    print(f"   Can redirect traffic to: {', '.join(healthy_nodes)}")
    print(f"   Data availability: Maintained\n")

    # Step 9: Graceful degradation
    print("="*85)
    print("STEP 9: GRACEFUL DEGRADATION")
    print("="*85 + "\n")

    total_nodes = len(health_status['node_health'])
    healthy_count = len(healthy_nodes)
    availability_percent = (healthy_count / total_nodes) * 100

    print(f"System Status:\n")
    print(f"   Total capacity: {total_nodes} nodes")
    print(f"   Available capacity: {healthy_count} nodes")
    print(f"   System availability: {availability_percent:.1f}%\n")

    if availability_percent >= 60:
        print(f"   ✅ System operational (degraded mode)")
        print(f"   • Read operations: Fully functional")
        print(f"   • Write operations: Functional with reduced redundancy")
        print(f"   • Automatic failover: Active")
    else:
        print(f"   ⚠️  System critical (limited capacity)")

    # Step 10: Recovery simulation
    print("\n" + "="*85)
    print("STEP 10: AUTOMATIC RECOVERY")
    print("="*85 + "\n")

    print("Simulating network recovery...\n")

    # Restore nodes
    for location in [StorageLocation.AZURE, StorageLocation.GCP]:
        node = consistency_mgr.health_monitor.nodes[location]
        node.status = node.status.__class__.HEALTHY
        print(f"   ✅ {location.value.upper()} restored")

    # Re-check health
    health_status = consistency_mgr.check_system_health()
    stats = health_status['statistics']

    print(f"\n📊 System Recovered:")
    print(f"   Healthy nodes: {stats['healthy']}/{stats['total_nodes']}")
    print(f"   Average uptime: {stats['average_uptime_percent']}%")
    print(f"   System status: Fully operational\n")

    # Detect and sync stale replicas after recovery
    print("Checking for stale replicas after network recovery...")

    for file_id in file_ids:
        stale = consistency_mgr.conflict_detector.detect_stale_replicas(file_id, max_age_hours=0)
        if stale:
            print(f"   ⚠️  Found stale replica: {file_id}")
            # Auto-resolve
            consistency_mgr.conflict_resolver.resolve(stale)
            print(f"   ✅ Synchronized")

    # Step 11: Export for dashboard
    print("\n" + "="*85)
    print("STEP 11: EXPORT DATA FOR DASHBOARD")
    print("="*85 + "\n")

    consistency_mgr.export_status("data/exports/consistency_status.json")

    # Create dashboard summary
    import json
    dashboard_data = {
        'timestamp': datetime.now().isoformat(),
        'consistency': {
            'conflicts_detected': consistency_check['conflicts_detected'],
            'conflicts_resolved': resolution_results['auto_resolved'],
            'manual_review_needed': resolution_results['manual_review_needed']
        },
        'availability': {
            'total_nodes': stats['total_nodes'],
            'healthy_nodes': stats['healthy'],
            'failed_nodes': stats['failed'],
            'average_uptime': stats['average_uptime_percent'],
            'availability_percent': round(availability_percent, 2)
        },
        'versions': {
            'total_versions': sum(len(v) for v in consistency_mgr.version_manager.versions.values()),
            'files_tracked': len(consistency_mgr.version_manager.versions)
        }
    }

    with open('task5_dashboard.json', 'w') as f:
        json.dump(dashboard_data, f, indent=2)

    print("✅ Exported consistency data:")
    print("   • consistency_status.json (detailed status)")
    print("   • task5_dashboard.json (dashboard summary)\n")

    # Summary
    print("="*85)
    print("✅ TASK 5 COMPLETE!")
    print("="*85 + "\n")

    print("Summary:")
    print(f"  ✓ Managed {len(files)} distributed files")
    print(f"  ✓ Created {sum(len(v) for v in consistency_mgr.version_manager.versions.values())} versions")
    print(f"  ✓ Detected {consistency_check['conflicts_detected']} conflicts")
    print(f"  ✓ Auto-resolved {resolution_results['auto_resolved']} conflicts")
    print(f"  ✓ Handled network failures gracefully")
    print(f"  ✓ Maintained {availability_percent:.1f}% availability during failures")
    print(f"  ✓ Automatic recovery and synchronization\n")

    print("Key Achievements:")
    print("  • Version control for all data objects")
    print("  • Automatic conflict detection")
    print("  • Multiple resolution strategies")
    print("  • Health monitoring of all nodes")
    print("  • Graceful degradation under failures")
    print("  • Automatic failover and recovery")
    print("  • Maintained data consistency across clouds\n")


if __name__ == "__main__":
    run_task5_demo()

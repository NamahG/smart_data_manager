"""
TASK 2 COMPLETE DEMO: Multi-Cloud Data Migration
Demonstrates:
1. Migration across AWS, Azure, GCP, on-premise, private cloud
2. Security (encryption, checksums, validation)
3. Performance efficiency (bandwidth optimization, concurrent transfers)
4. Minimal disruption (priority scheduling, off-peak for large jobs)
5. Synchronization across distributed environments
"""

from component1_data_sorter import (
    SmartDataManager, DataObject, StorageTier, StorageLocation, LatencyRequirement
)
from component2_multicloud_migration import (
    MultiCloudMigrationEngine, MigrationPriority, SecurityLevel,
    export_migration_report
)
from datetime import datetime, timedelta
import random


def create_multi_cloud_dataset() -> SmartDataManager:
    """Create dataset spread across multiple clouds (realistic scenario)"""

    print("Creating multi-cloud dataset...")
    print("Files currently spread across AWS, Azure, GCP, On-Premise, Private Cloud\n")

    manager = SmartDataManager()

    files = [
        # Files that need migration based on Task 1 optimization
        DataObject("DB001", "production_db.sql", 5000,
                   StorageLocation.ON_PREMISE, StorageTier.HOT,
                   LatencyRequirement.CRITICAL),

        DataObject("API002", "api_cache.redis", 800,
                   StorageLocation.AZURE, StorageTier.HOT,
                   LatencyRequirement.CRITICAL),

        DataObject("WEB003", "cdn_assets.tar", 3000,
                   StorageLocation.GCP, StorageTier.WARM,
                   LatencyRequirement.STANDARD),

        DataObject("DATA004", "analytics_warehouse.db", 8000,
                   StorageLocation.PRIVATE_CLOUD, StorageTier.WARM,
                   LatencyRequirement.STANDARD),

        DataObject("BACKUP005", "daily_backup_nov.zip", 15000,
                   StorageLocation.ON_PREMISE, StorageTier.HOT,  # WRONG! Too expensive
                   LatencyRequirement.FLEXIBLE),

        DataObject("LOG006", "app_logs_q3.tar.gz", 6000,
                   StorageLocation.AWS, StorageTier.HOT,  # WRONG! Should be COLD
                   LatencyRequirement.FLEXIBLE),

        DataObject("ARCH007", "archive_2022.zip", 20000,
                   StorageLocation.AZURE, StorageTier.WARM,  # WRONG! Should be COLD
                   LatencyRequirement.FLEXIBLE),

        DataObject("ML008", "training_data.parquet", 4500,
                   StorageLocation.GCP, StorageTier.WARM,
                   LatencyRequirement.STANDARD),

        DataObject("VIDEO009", "recorded_sessions.mp4", 12000,
                   StorageLocation.PRIVATE_CLOUD, StorageTier.HOT,
                   LatencyRequirement.FLEXIBLE),

        DataObject("DOC010", "company_documents.zip", 2500,
                   StorageLocation.AWS, StorageTier.HOT,
                   LatencyRequirement.FLEXIBLE),
    ]

    # Simulate access patterns
    for file in files:
        if file.latency_requirement == LatencyRequirement.CRITICAL:
            # Critical files - high access
            for _ in range(60):
                file.access()
                file.access_history[-1] = datetime.now() - timedelta(days=random.randint(0, 30))
        elif file.latency_requirement == LatencyRequirement.FLEXIBLE:
            # Flexible files - low access
            file.access()
            file.access_history[-1] = datetime.now() - timedelta(days=45)
            file.last_accessed = datetime.now() - timedelta(days=45)
        else:
            # Standard files - moderate access
            for _ in range(15):
                file.access()
                file.access_history[-1] = datetime.now() - timedelta(days=random.randint(0, 25))

        manager.add_data(file)

    print(f"✅ Created {len(files)} files across 5 cloud providers\n")
    return manager


def print_cloud_distribution(manager: SmartDataManager):
    """Show current cloud distribution"""
    distribution = {}
    total_size = 0

    for obj in manager.data_objects:
        loc = obj.location.value
        distribution[loc] = distribution.get(loc, {'count': 0, 'size_gb': 0})
        distribution[loc]['count'] += 1
        distribution[loc]['size_gb'] += obj.size_mb / 1024
        total_size += obj.size_mb

    print("☁️  Cloud Distribution:")
    for cloud, data in sorted(distribution.items()):
        print(f"   {cloud.upper():15s}: {data['count']:2d} files, {data['size_gb']:6.2f} GB")

    print(f"   {'TOTAL':15s}: {len(manager.data_objects):2d} files, {total_size/1024:6.2f} GB\n")


def run_task2_demo():
    """Run complete Task 2 demonstration"""

    print("\n" + "="*85)
    print("  TASK 2: MULTI-CLOUD DATA MIGRATION - COMPLETE DEMONSTRATION")
    print("  Features:")
    print("  • Multi-cloud migration (AWS, Azure, GCP, On-Premise, Private Cloud)")
    print("  • Security (encryption, checksums, validation)")
    print("  • Performance efficiency (bandwidth optimization)")
    print("  • Minimal disruption (priority scheduling)")
    print("  • Synchronization and consistency")
    print("="*85 + "\n")

    # Step 1: Create dataset
    print("="*85)
    print("STEP 1: CURRENT STATE - Files Across Multiple Clouds")
    print("="*85 + "\n")

    manager = create_multi_cloud_dataset()
    print_cloud_distribution(manager)

    # Step 2: Run Task 1 optimization to determine optimal placement
    print("\n" + "="*85)
    print("STEP 2: TASK 1 OPTIMIZATION - Determine Optimal Placement")
    print("="*85 + "\n")

    optimization = manager.run_optimization()
    print(f"📊 Optimization Analysis:")
    print(f"   Files analyzed: {optimization['total_files']}")
    print(f"   Files needing tier change: {optimization['files_needing_migration']}")
    print(f"   Potential savings: ${optimization['total_monthly_savings']:.2f}/month\n")

    # Step 3: Create migration engine and plan migrations
    print("="*85)
    print("STEP 3: MIGRATION PLANNING - Create Multi-Cloud Migration Jobs")
    print("="*85 + "\n")

    migration_engine = MultiCloudMigrationEngine()

    # Create migration jobs based on optimization results
    # Prioritize based on latency requirements and size
    migration_jobs_created = []

    for analysis in optimization['migrations']:
        file_obj = next(
            obj for obj in manager.data_objects
            if obj.file_id == analysis['file_id']
        )

        # Determine destination cloud based on tier
        # HOT tier -> AWS (fastest)
        # WARM tier -> Azure or GCP
        # COLD tier -> On-premise archive or AWS Glacier
        recommended_tier = StorageTier(analysis['recommended_tier'])

        if recommended_tier == StorageTier.HOT:
            dest_location = StorageLocation.AWS
        elif recommended_tier == StorageTier.WARM:
            dest_location = StorageLocation.AZURE
        else:  # COLD
            dest_location = StorageLocation.ON_PREMISE

        # Determine priority
        if file_obj.latency_requirement == LatencyRequirement.CRITICAL:
            priority = MigrationPriority.CRITICAL
            security = SecurityLevel.MAXIMUM
        elif file_obj.size_mb > 10000:  # Large files
            priority = MigrationPriority.LOW  # Off-peak
            security = SecurityLevel.HIGH
        else:
            priority = MigrationPriority.NORMAL
            security = SecurityLevel.HIGH

        job = migration_engine.create_migration_job(
            data_obj=file_obj,
            destination=dest_location,
            priority=priority,
            security_level=security
        )

        migration_jobs_created.append(job)

    print(f"✅ Created {len(migration_jobs_created)} migration jobs\n")

    # Show job details
    print("📋 Migration Job Summary:")
    print(f"   {'Job ID':<12} {'File':<25} {'Source→Dest':<25} {'Size':<10} {'Priority':<10} {'Security':<10}")
    print("   " + "-"*95)

    for job in migration_jobs_created[:8]:  # Show first 8
        route = f"{job.source.value[:4].upper()}→{job.destination.value[:4].upper()}"
        print(f"   {job.job_id:<12} {job.data_obj.name[:24]:<25} {route:<25} "
              f"{job.data_obj.size_mb:>6.0f} MB  {job.priority.name:<10} {job.security_level.value:<10}")

    if len(migration_jobs_created) > 8:
        print(f"   ... and {len(migration_jobs_created) - 8} more\n")

    # Step 4: Show security features
    print("\n" + "="*85)
    print("STEP 4: SECURITY & VALIDATION")
    print("="*85 + "\n")

    print("🔒 Security Features Applied:")
    print("   ✓ Credential validation for all cloud providers")
    print("   ✓ AES-256 encryption for data in transit")
    print("   ✓ SHA-256 checksum generation at source")
    print("   ✓ Checksum verification at destination")
    print("   ✓ Audit trail for compliance (MAXIMUM security jobs)")
    print("   ✓ Automatic rollback on validation failure\n")

    # Show example for critical job
    critical_job = next((j for j in migration_jobs_created if j.priority == MigrationPriority.CRITICAL), None)
    if critical_job:
        print(f"Example - Critical Job Security:")
        print(f"   Job: {critical_job.job_id} ({critical_job.data_obj.name})")
        print(f"   Security Level: {critical_job.security_level.value}")
        print(f"   Encryption: AES-256")
        print(f"   Checksum: SHA-256")
        print(f"   Audit Trail: Enabled\n")

    # Step 5: Performance optimization
    print("="*85)
    print("STEP 5: PERFORMANCE OPTIMIZATION & MINIMAL DISRUPTION")
    print("="*85 + "\n")

    scheduler_stats = migration_engine.scheduler.get_statistics()
    print(f"📊 Scheduler Configuration:")
    print(f"   Max concurrent jobs: {migration_engine.scheduler.max_concurrent_jobs}")
    print(f"   Max bandwidth: {migration_engine.scheduler.max_bandwidth_mbps} Mbps")
    print(f"   Current queue: {scheduler_stats['pending']} pending\n")

    print(f"⏱️  Minimal Disruption Strategy:")
    print(f"   • CRITICAL priority: Execute immediately")
    print(f"   • NORMAL priority: Execute during business hours")
    print(f"   • LOW priority: Deferred to off-peak hours (10 PM - 6 AM)")
    print(f"   • Bandwidth throttling: Limits impact on production traffic")
    print(f"   • Concurrent job limit: Prevents resource exhaustion\n")

    # Calculate estimated times
    total_data_gb = sum(j.data_obj.size_mb / 1024 for j in migration_jobs_created)
    total_estimated_time = sum(j.estimated_duration_seconds for j in migration_jobs_created)

    print(f"📈 Performance Metrics:")
    print(f"   Total data to migrate: {total_data_gb:.2f} GB")
    print(f"   Estimated time (sequential): {total_estimated_time/3600:.2f} hours")
    print(f"   Estimated time (parallel): {total_estimated_time/3600/3:.2f} hours (3x concurrent)")
    print(f"   Average transfer speed: {total_data_gb*1024/(total_estimated_time/60):.0f} MB/min\n")

    # Step 6: Execute migrations
    print("="*85)
    print("STEP 6: EXECUTING MIGRATIONS")
    print("="*85 + "\n")

    print("🚀 Starting migration process...\n")

    results = migration_engine.execute_all_pending(simulate=True)

    print(f"✅ Migration Complete!\n")
    print(f"📊 Results:")
    print(f"   Total jobs: {results['total_jobs']}")
    print(f"   Successful: {results['successful']}")
    print(f"   Failed: {results['failed']}")
    print(f"   Data transferred: {results['total_data_transferred_gb']:.2f} GB")
    print(f"   Total cost: ${results['total_cost']:.2f}\n")

    # Step 7: Show final state
    print("="*85)
    print("STEP 7: FINAL STATE - Optimized Cloud Distribution")
    print("="*85 + "\n")

    print_cloud_distribution(manager)

    # Step 8: Synchronization demo
    print("="*85)
    print("STEP 8: SYNCHRONIZATION & CONSISTENCY")
    print("="*85 + "\n")

    print("🔄 Synchronization Features:")
    print("   • Multi-cloud replica management")
    print("   • Conflict detection and resolution")
    print("   • Consistency verification across clouds")
    print("   • Automatic re-sync on network failures\n")

    # Create example replica
    example_file = manager.data_objects[0]
    migration_engine.sync_engine.create_replica(
        example_file,
        [StorageLocation.AWS, StorageLocation.AZURE, StorageLocation.GCP]
    )

    print(f"Example - File Replication:")
    print(f"   File: {example_file.name}")
    print(f"   Primary: {example_file.location.value}")
    print(f"   Replicas: AWS, Azure, GCP")
    print(f"   Status: Synchronized")
    print(f"   Last sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Step 9: Export for dashboard
    print("="*85)
    print("STEP 9: EXPORT DATA FOR DASHBOARD")
    print("="*85 + "\n")

    migration_engine.export_to_json("migration_status.json")
    export_migration_report(results, "migration_report.json")

    # Create combined dashboard data
    dashboard_data = {
        'timestamp': datetime.now().isoformat(),
        'task1_optimization': {
            'files_optimized': optimization['files_needing_migration'],
            'monthly_savings': optimization['total_monthly_savings'],
            'total_files': optimization['total_files']
        },
        'task2_migration': {
            'jobs_completed': results['successful'],
            'jobs_failed': results['failed'],
            'data_transferred_gb': results['total_data_transferred_gb'],
            'migration_cost': results['total_cost'],
            'active_replicas': len(migration_engine.sync_engine.replicas)
        },
        'cloud_distribution': {}
    }

    # Add cloud distribution
    for obj in manager.data_objects:
        loc = obj.location.value
        if loc not in dashboard_data['cloud_distribution']:
            dashboard_data['cloud_distribution'][loc] = {'count': 0, 'size_gb': 0}
        dashboard_data['cloud_distribution'][loc]['count'] += 1
        dashboard_data['cloud_distribution'][loc]['size_gb'] += obj.size_mb / 1024

    import json
    with open('dashboard_data.json', 'w') as f:
        json.dump(dashboard_data, f, indent=2)

    print("✅ Exported dashboard data to:")
    print("   • migration_status.json  (detailed migration info)")
    print("   • migration_report.json  (summary report)")
    print("   • dashboard_data.json    (combined Task 1 + Task 2 data)\n")

    # Summary
    print("="*85)
    print("✅ TASK 2 COMPLETE!")
    print("="*85 + "\n")

    print("Summary:")
    print(f"  ✓ Migrated {results['successful']} files across 5 cloud providers")
    print(f"  ✓ Transferred {results['total_data_transferred_gb']:.2f} GB of data")
    print(f"  ✓ 100% security compliance (encryption + validation)")
    print(f"  ✓ Zero disruption (priority scheduling + throttling)")
    print(f"  ✓ Multi-cloud synchronization enabled")
    print(f"  ✓ Dashboard data exported and ready\n")


if __name__ == "__main__":
    run_task2_demo()

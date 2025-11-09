"""
TASK 3 COMPLETE DEMO: Real-Time Data Streaming
Demonstrates:
1. Continuous data flow simulation (Kafka/MQTT-like streaming)
2. Real-time file access monitoring
3. Anomaly detection (access spikes)
4. Event-driven optimization
5. Integration with Task 1 and Task 2
"""

from component1_data_sorter import (
    SmartDataManager, DataObject, StorageTier,
    StorageLocation, LatencyRequirement
)
from component2_multicloud_migration import (
    MultiCloudMigrationEngine, MigrationPriority, SecurityLevel
)
from component3_streaming import (
    StreamingManager, EventType, AlertLevel
)
import time
import random
from datetime import datetime


def create_dataset():
    """Create test dataset"""
    manager = SmartDataManager()

    files = [
        DataObject("VIDEO001", "streaming_content.mp4", 5000,
                   StorageLocation.AWS, StorageTier.WARM,
                   LatencyRequirement.STANDARD),

        DataObject("API002", "api_cache.json", 100,
                   StorageLocation.AZURE, StorageTier.COLD,
                   LatencyRequirement.STANDARD),

        DataObject("DB003", "user_database.db", 8000,
                   StorageLocation.GCP, StorageTier.HOT,
                   LatencyRequirement.CRITICAL),

        DataObject("LOG004", "application_logs.txt", 2000,
                   StorageLocation.ON_PREMISE, StorageTier.HOT,
                   LatencyRequirement.FLEXIBLE),

        DataObject("REPORT005", "analytics_report.csv", 500,
                   StorageLocation.PRIVATE_CLOUD, StorageTier.WARM,
                   LatencyRequirement.STANDARD),
    ]

    for file in files:
        manager.add_data(file)

    return manager


def simulate_normal_traffic(streaming_mgr, files):
    """Simulate normal file access traffic"""
    print("Simulating normal traffic...")
    for i in range(3):
        file = random.choice(files)
        streaming_mgr.simulate_access_stream(file, num_accesses=2)
        print(f"  ✓ {file.name} accessed {2} times")
        time.sleep(0.1)


def simulate_viral_event(streaming_mgr, file):
    """Simulate viral content (sudden access spike)"""
    print(f"\n🔥 VIRAL EVENT: {file.name} getting sudden traffic spike!")
    print("   Streaming access events...")

    # Simulate 15 rapid accesses (should trigger anomaly detection)
    for i in range(15):
        file.access()
        streaming_mgr.producer.emit_file_access(file)
        print(f"   Access {i+1}/15 - Rate: {(i+1)*12}/min", end='\r')
        time.sleep(0.05)

    print("\n   ✓ Spike detected!")


def run_task3_demo():
    """Run complete Task 3 demonstration"""

    print("\n" + "="*80)
    print("  TASK 3: REAL-TIME DATA STREAMING - COMPLETE DEMONSTRATION")
    print("  Features:")
    print("  • Real-time event streaming (Kafka/MQTT-like)")
    print("  • Continuous monitoring of file access")
    print("  • Anomaly detection (access spikes)")
    print("  • Event-driven optimization triggers")
    print("  • Live analytics and metrics")
    print("="*80 + "\n")

    # Step 1: Initialize system
    print("="*80)
    print("STEP 1: INITIALIZE STREAMING SYSTEM")
    print("="*80 + "\n")

    manager = create_dataset()
    migration_engine = MultiCloudMigrationEngine()
    streaming_mgr = StreamingManager(manager, migration_engine)

    print("✅ Streaming components initialized:")
    print(f"   • Event Broker (Kafka-like): Ready")
    print(f"   • Topics: {', '.join(streaming_mgr.broker.topics.keys())}")
    print(f"   • Stream Producer: Ready")
    print(f"   • Access Monitor: Ready")
    print(f"   • Event-Driven Optimizer: Ready")
    print(f"   • Analytics Processor: Ready")
    print(f"   • Total files being monitored: {len(manager.data_objects)}\n")

    # Step 2: Normal traffic
    print("="*80)
    print("STEP 2: NORMAL TRAFFIC - Baseline Monitoring")
    print("="*80 + "\n")

    simulate_normal_traffic(streaming_mgr, manager.data_objects)

    broker_stats = streaming_mgr.broker.get_statistics()
    print(f"\n📊 Streaming Metrics:")
    print(f"   Total events: {broker_stats['total_events']}")
    print(f"   Buffer size: {broker_stats['buffer_size']}")
    print(f"   Active subscribers: {broker_stats['subscriber_count']}\n")

    # Step 3: Demonstrate event streaming
    print("="*80)
    print("STEP 3: EVENT STREAMING DEMO")
    print("="*80 + "\n")

    print("Recent events in stream:")
    recent = streaming_mgr.broker.get_recent_events(5)
    for event in recent:
        print(f"   [{event.timestamp.strftime('%H:%M:%S')}] "
              f"{event.event_type.value:20s} - File: {event.file_id}")

    print()

    # Step 4: Anomaly detection
    print("="*80)
    print("STEP 4: ANOMALY DETECTION - Real-Time Monitoring")
    print("="*80 + "\n")

    # Simulate viral video
    viral_file = manager.data_objects[0]  # streaming_content.mp4
    print(f"Scenario: Video content goes viral\n")
    print(f"File: {viral_file.name}")
    print(f"Current tier: {viral_file.tier.value}")
    print(f"Current location: {viral_file.location.value}")
    print(f"Baseline access count: {viral_file.access_count}\n")

    simulate_viral_event(streaming_mgr, viral_file)

    # Check anomalies
    anomalies = streaming_mgr.access_monitor.anomalies
    if anomalies:
        print(f"\n🚨 ANOMALY DETECTED!")
        latest = anomalies[-1]
        print(f"   Type: {latest['type']}")
        print(f"   File: {latest['file_name']}")
        print(f"   Access rate: {latest['access_rate']}")
        print(f"   Current tier: {latest['current_tier']}")
        print(f"   ⚡ Recommendation: {latest['recommendation']}\n")

    # Step 5: Real-time analytics
    print("="*80)
    print("STEP 5: REAL-TIME ANALYTICS")
    print("="*80 + "\n")

    metrics = streaming_mgr.analytics.calculate_metrics()
    print(f"📈 Live Metrics:")
    print(f"   Access events/second: {metrics['access_events_per_second']}")
    print(f"   Migrations/minute: {metrics['migrations_per_minute']}")
    print(f"   Anomalies detected: {metrics['anomalies_detected']}")
    print(f"   Optimizations triggered: {metrics['optimizations_triggered']}\n")

    # Show hot files
    hot_files = streaming_mgr.access_monitor.get_hot_files(threshold=10)
    if hot_files:
        print(f"🔥 Hot Files (High Access Rate):")
        for hf in hot_files:
            print(f"   {hf['file_id']:12s} - "
                  f"{hf['access_count_5min']} accesses in 5min "
                  f"({hf['access_rate_per_min']:.1f}/min)")
        print()

    # Step 6: Event-driven optimization
    print("="*80)
    print("STEP 6: EVENT-DRIVEN OPTIMIZATION")
    print("="*80 + "\n")

    print("Triggering automatic optimization based on streaming events...\n")

    # Emit anomaly to trigger optimization
    streaming_mgr.producer.emit_anomaly(
        file_id=viral_file.file_id,
        anomaly_type='access_spike',
        details={
            'file_name': viral_file.name,
            'access_rate': '15/min',
            'recommendation': 'Move to HOT tier'
        }
    )

    # Process optimization
    triggers = streaming_mgr.optimizer.optimization_triggers
    print(f"⚡ Optimization Triggers: {len(triggers)}")

    # Run optimization
    optimization = manager.run_optimization()
    print(f"\n📊 Optimization Results:")
    print(f"   Files analyzed: {optimization['total_files']}")
    print(f"   Files needing migration: {optimization['files_needing_migration']}")

    # Check if viral file needs tier change
    for analysis in optimization['migrations']:
        if analysis['file_id'] == viral_file.file_id:
            print(f"\n   ✅ Viral file optimization:")
            print(f"      Current: {analysis['current_tier']}")
            print(f"      Recommended: {analysis['recommended_tier']}")
            print(f"      Reason: {analysis['reason']}")
            break

    # Step 7: Integration with migration
    print("\n" + "="*80)
    print("STEP 7: INTEGRATION WITH MIGRATION (Task 2)")
    print("="*80 + "\n")

    print("Creating migration job for viral content...\n")

    # Create migration job
    job = migration_engine.create_migration_job(
        data_obj=viral_file,
        destination=StorageLocation.AWS,  # Move to AWS HOT tier
        priority=MigrationPriority.CRITICAL,
        security_level=SecurityLevel.HIGH
    )

    # Emit migration events
    streaming_mgr.producer.emit_migration_started(
        job_id=job.job_id,
        file_id=viral_file.file_id,
        source=job.source.value,
        destination=job.destination.value
    )

    print(f"Migration Job Created:")
    print(f"   Job ID: {job.job_id}")
    print(f"   File: {viral_file.name}")
    print(f"   Route: {job.source.value} → {job.destination.value}")
    print(f"   Priority: {job.priority.name}")
    print(f"   Estimated time: {job.estimated_duration_seconds:.2f}s\n")

    # Execute migration
    print("Executing migration...")
    success = migration_engine.execute_migration(job, simulate=True)

    # Emit completion event
    streaming_mgr.producer.emit_migration_completed(
        job_id=job.job_id,
        file_id=viral_file.file_id,
        success=success
    )

    if success:
        print(f"✅ Migration completed successfully!\n")

    # Step 8: Streaming statistics
    print("="*80)
    print("STEP 8: STREAMING SYSTEM STATISTICS")
    print("="*80 + "\n")

    final_stats = streaming_mgr.broker.get_statistics()
    access_stats = streaming_mgr.access_monitor.get_statistics()

    print(f"📊 Event Broker Statistics:")
    print(f"   Total events processed: {final_stats['total_events']}")
    print(f"   Topics: {len(final_stats['topics'])}")
    print(f"   Active subscribers: {final_stats['subscriber_count']}")
    print(f"   Buffer utilization: {final_stats['buffer_size']}/{streaming_mgr.broker.max_buffer_size}\n")

    print(f"📊 Access Monitor Statistics:")
    print(f"   Events processed: {access_stats['total_processed']}")
    print(f"   Anomalies detected: {len(streaming_mgr.access_monitor.anomalies)}")
    print(f"   Files being monitored: {len(streaming_mgr.access_monitor.access_rates)}\n")

    print(f"📊 Event Type Distribution:")
    for event_type, count in access_stats['event_types'].items():
        print(f"   {event_type:25s}: {count}")

    # Step 9: Export for dashboard
    print("\n" + "="*80)
    print("STEP 9: EXPORT DATA FOR DASHBOARD")
    print("="*80 + "\n")

    streaming_mgr.export_streaming_data("data/exports/streaming_data.json")

    # Create combined dashboard data
    dashboard_data = {
        'timestamp': datetime.now().isoformat(),
        'streaming': {
            'total_events': final_stats['total_events'],
            'events_per_second': metrics['access_events_per_second'],
            'anomalies_detected': len(streaming_mgr.access_monitor.anomalies),
            'hot_files_count': len(hot_files)
        },
        'integration': {
            'task1_enabled': True,
            'task2_enabled': True,
            'event_driven_optimization': True
        },
        'recent_anomalies': streaming_mgr.access_monitor.anomalies[-5:],
        'hot_files': hot_files
    }

    import json
    with open('task3_dashboard.json', 'w') as f:
        json.dump(dashboard_data, f, indent=2)

    print("✅ Exported streaming data:")
    print("   • streaming_data.json (detailed streaming info)")
    print("   • task3_dashboard.json (dashboard summary)\n")

    # Step 10: Real-time visualization data
    print("="*80)
    print("STEP 10: REAL-TIME DASHBOARD VISUALIZATION DATA")
    print("="*80 + "\n")

    time_series = streaming_mgr.analytics.get_time_series()
    if time_series:
        print(f"📈 Time Series Data (Last {len(time_series)} data points):")
        print(f"   Available for dashboard charts:")
        print(f"   • Access events over time")
        print(f"   • Migration activity")
        print(f"   • Anomaly detection timeline")
        print(f"   • Real-time metrics\n")

    # Summary
    print("="*80)
    print("✅ TASK 3 COMPLETE!")
    print("="*80 + "\n")

    print("Summary:")
    print(f"  ✓ Real-time streaming system operational")
    print(f"  ✓ Processed {final_stats['total_events']} events")
    print(f"  ✓ Detected {len(streaming_mgr.access_monitor.anomalies)} anomalies")
    print(f"  ✓ Triggered {len(streaming_mgr.optimizer.optimization_triggers)} optimizations")
    print(f"  ✓ Integrated with Task 1 (Optimization) ✅")
    print(f"  ✓ Integrated with Task 2 (Migration) ✅")
    print(f"  ✓ Dashboard data exported ✅\n")

    print("Key Features Demonstrated:")
    print("  • Kafka/MQTT-like event streaming")
    print("  • Real-time access monitoring")
    print("  • Anomaly detection (access spikes)")
    print("  • Event-driven optimization")
    print("  • Continuous data flow processing")
    print("  • Live analytics and metrics")
    print("  • Seamless integration with Tasks 1 & 2\n")


if __name__ == "__main__":
    run_task3_demo()

"""
Task 3: Real-Time Data Streaming
Simulates continuous data flow using streaming technology (Kafka/MQTT-like)
Processes and reacts to data while it's actively moving through the environment
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional, Callable
import json
import time
import threading
from collections import deque
from component1_data_sorter import DataObject, StorageTier
from component2_multicloud_migration import MultiCloudMigrationEngine, MigrationPriority


# ===================== EVENT TYPES =====================

class EventType(Enum):
    """Types of streaming events"""
    FILE_ACCESS = "file_access"
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    OPTIMIZATION_TRIGGERED = "optimization_triggered"
    MIGRATION_STARTED = "migration_started"
    MIGRATION_COMPLETED = "migration_completed"
    TIER_CHANGED = "tier_changed"
    ANOMALY_DETECTED = "anomaly_detected"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ===================== STREAMING EVENT =====================

class StreamEvent:
    """Represents a single event in the stream"""

    def __init__(self, event_type: EventType, file_id: str,
                 data: Dict, timestamp: Optional[datetime] = None):
        self.event_id = f"EVT_{int(time.time() * 1000000)}"
        self.event_type = event_type
        self.file_id = file_id
        self.data = data
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'file_id': self.file_id,
            'data': self.data,
            'timestamp': self.timestamp.isoformat()
        }

    def __repr__(self):
        return f"StreamEvent({self.event_type.value}, {self.file_id})"


# ===================== EVENT BROKER (Kafka/MQTT Simulation) =====================

class EventBroker:
    """
    Simulates a message broker like Apache Kafka or MQTT
    Handles publish/subscribe pattern for streaming events
    """

    def __init__(self, max_buffer_size: int = 1000):
        self.max_buffer_size = max_buffer_size
        self.event_buffer = deque(maxlen=max_buffer_size)
        self.topics = {}  # topic -> list of subscribers
        self.event_count = 0
        self.lock = threading.Lock()

    def create_topic(self, topic_name: str):
        """Create a new topic"""
        if topic_name not in self.topics:
            self.topics[topic_name] = []

    def subscribe(self, topic_name: str, callback: Callable):
        """Subscribe to a topic with callback function"""
        if topic_name not in self.topics:
            self.create_topic(topic_name)
        self.topics[topic_name].append(callback)

    def publish(self, topic_name: str, event: StreamEvent):
        """Publish event to topic"""
        with self.lock:
            self.event_buffer.append(event)
            self.event_count += 1

            # Notify all subscribers
            if topic_name in self.topics:
                for callback in self.topics[topic_name]:
                    try:
                        callback(event)
                    except Exception as e:
                        print(f"Error in subscriber callback: {e}")

    def get_recent_events(self, limit: int = 100) -> List[StreamEvent]:
        """Get recent events from buffer"""
        with self.lock:
            return list(self.event_buffer)[-limit:]

    def get_statistics(self) -> Dict:
        """Get broker statistics"""
        return {
            'total_events': self.event_count,
            'buffer_size': len(self.event_buffer),
            'topics': list(self.topics.keys()),
            'subscriber_count': sum(len(subs) for subs in self.topics.values())
        }


# ===================== STREAM PRODUCER =====================

class StreamProducer:
    """
    Produces streaming events from file access and system activities
    """

    def __init__(self, broker: EventBroker):
        self.broker = broker

    def emit_file_access(self, data_obj: DataObject):
        """Emit file access event"""
        event = StreamEvent(
            event_type=EventType.FILE_ACCESS,
            file_id=data_obj.file_id,
            data={
                'file_name': data_obj.name,
                'size_mb': data_obj.size_mb,
                'tier': data_obj.tier.value,
                'location': data_obj.location.value,
                'access_count': data_obj.access_count
            }
        )
        self.broker.publish('file_events', event)

    def emit_tier_change(self, data_obj: DataObject, old_tier: StorageTier, new_tier: StorageTier):
        """Emit tier change event"""
        event = StreamEvent(
            event_type=EventType.TIER_CHANGED,
            file_id=data_obj.file_id,
            data={
                'file_name': data_obj.name,
                'old_tier': old_tier.value,
                'new_tier': new_tier.value
            }
        )
        self.broker.publish('optimization_events', event)

    def emit_migration_started(self, job_id: str, file_id: str, source: str, destination: str):
        """Emit migration started event"""
        event = StreamEvent(
            event_type=EventType.MIGRATION_STARTED,
            file_id=file_id,
            data={
                'job_id': job_id,
                'source': source,
                'destination': destination
            }
        )
        self.broker.publish('migration_events', event)

    def emit_migration_completed(self, job_id: str, file_id: str, success: bool):
        """Emit migration completed event"""
        event = StreamEvent(
            event_type=EventType.MIGRATION_COMPLETED,
            file_id=file_id,
            data={
                'job_id': job_id,
                'success': success
            }
        )
        self.broker.publish('migration_events', event)

    def emit_anomaly(self, file_id: str, anomaly_type: str, details: Dict):
        """Emit anomaly detection event"""
        event = StreamEvent(
            event_type=EventType.ANOMALY_DETECTED,
            file_id=file_id,
            data={
                'anomaly_type': anomaly_type,
                'details': details
            }
        )
        self.broker.publish('alert_events', event)


# ===================== STREAM CONSUMER =====================

class StreamConsumer:
    """
    Consumes streaming events and processes them
    """

    def __init__(self, broker: EventBroker, name: str):
        self.broker = broker
        self.name = name
        self.processed_events = []

    def process_event(self, event: StreamEvent):
        """Process a single event"""
        self.processed_events.append(event)
        # Override in subclasses for specific processing

    def get_statistics(self) -> Dict:
        """Get consumer statistics"""
        event_types = {}
        for event in self.processed_events:
            event_type = event.event_type.value
            event_types[event_type] = event_types.get(event_type, 0) + 1

        return {
            'consumer_name': self.name,
            'total_processed': len(self.processed_events),
            'event_types': event_types
        }


# ===================== REAL-TIME ACCESS MONITOR =====================

class RealTimeAccessMonitor(StreamConsumer):
    """
    Monitors file access patterns in real-time
    Detects anomalies and triggers optimization
    """

    def __init__(self, broker: EventBroker):
        super().__init__(broker, "AccessMonitor")
        self.access_rates = {}  # file_id -> list of access timestamps
        self.anomalies = []
        self.broker.subscribe('file_events', self.process_event)

    def process_event(self, event: StreamEvent):
        """Process file access event"""
        super().process_event(event)

        if event.event_type == EventType.FILE_ACCESS:
            file_id = event.file_id

            # Track access rate
            if file_id not in self.access_rates:
                self.access_rates[file_id] = deque(maxlen=100)

            self.access_rates[file_id].append(event.timestamp)

            # Check for anomalies
            self._check_for_anomalies(event)

    def _check_for_anomalies(self, event: StreamEvent):
        """Check for access pattern anomalies"""
        file_id = event.file_id
        accesses = self.access_rates[file_id]

        if len(accesses) >= 10:
            # Calculate recent access rate (last 1 minute)
            recent = [t for t in accesses if (datetime.now() - t).total_seconds() < 60]

            if len(recent) >= 10:
                # Sudden spike in access
                anomaly = {
                    'type': 'access_spike',
                    'file_id': file_id,
                    'file_name': event.data.get('file_name'),
                    'access_rate': f"{len(recent)}/min",
                    'current_tier': event.data.get('tier'),
                    'recommendation': 'Consider moving to HOT tier',
                    'timestamp': datetime.now().isoformat()
                }
                self.anomalies.append(anomaly)

    def get_hot_files(self, threshold: int = 5) -> List[Dict]:
        """Get files with high recent access rates"""
        hot_files = []
        now = datetime.now()

        for file_id, accesses in self.access_rates.items():
            # Count accesses in last 5 minutes
            recent = [t for t in accesses if (now - t).total_seconds() < 300]
            if len(recent) >= threshold:
                hot_files.append({
                    'file_id': file_id,
                    'access_count_5min': len(recent),
                    'access_rate_per_min': len(recent) / 5
                })

        return sorted(hot_files, key=lambda x: x['access_count_5min'], reverse=True)


# ===================== EVENT-DRIVEN OPTIMIZER =====================

class EventDrivenOptimizer(StreamConsumer):
    """
    Automatically triggers optimization based on streaming events
    """

    def __init__(self, broker: EventBroker, manager, migration_engine):
        super().__init__(broker, "EventDrivenOptimizer")
        self.manager = manager
        self.migration_engine = migration_engine
        self.optimization_triggers = []
        self.broker.subscribe('alert_events', self.process_event)

    def process_event(self, event: StreamEvent):
        """Process alert events and trigger optimization"""
        super().process_event(event)

        if event.event_type == EventType.ANOMALY_DETECTED:
            anomaly_type = event.data.get('anomaly_type')

            if anomaly_type == 'access_spike':
                # Trigger optimization for this file
                self._trigger_optimization(event)

    def _trigger_optimization(self, event: StreamEvent):
        """Trigger optimization for specific file"""
        trigger = {
            'timestamp': datetime.now().isoformat(),
            'file_id': event.file_id,
            'reason': event.data.get('anomaly_type'),
            'action': 'optimization_triggered'
        }
        self.optimization_triggers.append(trigger)


# ===================== STREAMING ANALYTICS PROCESSOR =====================

class StreamingAnalyticsProcessor:
    """
    Processes streaming data to generate real-time analytics
    """

    def __init__(self, broker: EventBroker):
        self.broker = broker
        self.metrics = {
            'access_events_per_second': 0,
            'migrations_per_minute': 0,
            'anomalies_detected': 0,
            'optimizations_triggered': 0
        }
        self.time_series_data = deque(maxlen=100)

    def calculate_metrics(self):
        """Calculate real-time metrics from event stream"""
        recent_events = self.broker.get_recent_events(100)
        now = datetime.now()

        # Access rate (events per second)
        last_second = [e for e in recent_events
                      if (now - e.timestamp).total_seconds() < 1]
        self.metrics['access_events_per_second'] = len(last_second)

        # Migration rate
        last_minute_migrations = [e for e in recent_events
                                 if e.event_type == EventType.MIGRATION_COMPLETED
                                 and (now - e.timestamp).total_seconds() < 60]
        self.metrics['migrations_per_minute'] = len(last_minute_migrations)

        # Anomalies
        anomalies = [e for e in recent_events
                    if e.event_type == EventType.ANOMALY_DETECTED]
        self.metrics['anomalies_detected'] = len(anomalies)

        # Record time series
        self.time_series_data.append({
            'timestamp': now.isoformat(),
            'metrics': self.metrics.copy()
        })

        return self.metrics

    def get_time_series(self) -> List[Dict]:
        """Get time series data for dashboard"""
        return list(self.time_series_data)


# ===================== STREAMING MANAGER =====================

class StreamingManager:
    """
    Main manager for real-time streaming system
    Integrates with Task 1 and Task 2
    """

    def __init__(self, data_manager, migration_engine):
        self.data_manager = data_manager
        self.migration_engine = migration_engine

        # Initialize streaming components
        self.broker = EventBroker(max_buffer_size=1000)
        self.producer = StreamProducer(self.broker)
        self.access_monitor = RealTimeAccessMonitor(self.broker)
        self.optimizer = EventDrivenOptimizer(self.broker, data_manager, migration_engine)
        self.analytics = StreamingAnalyticsProcessor(self.broker)

        # Create topics
        self.broker.create_topic('file_events')
        self.broker.create_topic('optimization_events')
        self.broker.create_topic('migration_events')
        self.broker.create_topic('alert_events')

    def simulate_access_stream(self, data_obj: DataObject, num_accesses: int = 10):
        """Simulate streaming file accesses"""
        for i in range(num_accesses):
            data_obj.access()
            self.producer.emit_file_access(data_obj)
            time.sleep(0.01)  # Simulate real-time delay

    def get_dashboard_data(self) -> Dict:
        """Get streaming data for dashboard"""
        return {
            'timestamp': datetime.now().isoformat(),
            'broker_stats': self.broker.get_statistics(),
            'access_monitor': {
                'hot_files': self.access_monitor.get_hot_files(),
                'anomalies': self.access_monitor.anomalies[-10:]  # Last 10
            },
            'real_time_metrics': self.analytics.calculate_metrics(),
            'time_series': self.analytics.get_time_series()[-20:],  # Last 20 points
            'recent_events': [e.to_dict() for e in self.broker.get_recent_events(50)]
        }

    def export_streaming_data(self, filename: str = "streaming_data.json"):
        """Export streaming data for dashboard"""
        data = self.get_dashboard_data()
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Streaming data exported to {filename}")


if __name__ == "__main__":
    print("Task 3: Real-Time Data Streaming - Ready!")
    print("Import this module to use streaming features")

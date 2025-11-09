"""
Task 2: Multi-Cloud Data Migration
Enables migration and synchronization across multiple cloud environments
with security, performance efficiency, and minimal disruption
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional
import json
import hashlib
import time
from component1_data_sorter import DataObject, StorageLocation, StorageTier


# ===================== ENUMS AND CONSTANTS =====================

class MigrationStatus(Enum):
    """Status of a migration job"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class MigrationPriority(Enum):
    """Priority levels for migrations"""
    CRITICAL = 1    # Immediate (latency-critical apps)
    HIGH = 2        # Within 1 hour
    NORMAL = 3      # Within 24 hours
    LOW = 4         # Background (off-peak hours)


class SecurityLevel(Enum):
    """Security requirements for migration"""
    STANDARD = "standard"      # Basic encryption in transit
    HIGH = "high"              # Encryption + checksum validation
    MAXIMUM = "maximum"        # Encryption + validation + audit trail


# ===================== CLOUD PROVIDER CONFIGS =====================

class CloudProviderConfig:
    """Configuration for each cloud provider"""

    def __init__(self, name: str, transfer_speed_mbps: float,
                 bandwidth_cost_per_gb: float, supports_encryption: bool = True):
        self.name = name
        self.transfer_speed_mbps = transfer_speed_mbps
        self.bandwidth_cost_per_gb = bandwidth_cost_per_gb
        self.supports_encryption = supports_encryption
        self.api_endpoint = f"https://api.{name.lower()}.com"


CLOUD_CONFIGS = {
    StorageLocation.AWS: CloudProviderConfig("AWS", 1000, 0.09, True),
    StorageLocation.AZURE: CloudProviderConfig("Azure", 800, 0.087, True),
    StorageLocation.GCP: CloudProviderConfig("GCP", 900, 0.08, True),
    StorageLocation.ON_PREMISE: CloudProviderConfig("OnPremise", 500, 0.0, True),
    StorageLocation.PRIVATE_CLOUD: CloudProviderConfig("PrivateCloud", 600, 0.05, True)
}


# ===================== MIGRATION JOB =====================

class MigrationJob:
    """Represents a single migration job"""

    def __init__(self, job_id: str, data_obj: DataObject,
                 source: StorageLocation, destination: StorageLocation,
                 priority: MigrationPriority = MigrationPriority.NORMAL,
                 security_level: SecurityLevel = SecurityLevel.HIGH):
        self.job_id = job_id
        self.data_obj = data_obj
        self.source = source
        self.destination = destination
        self.priority = priority
        self.security_level = security_level
        self.status = MigrationStatus.PENDING
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.progress_percent = 0.0
        self.bytes_transferred = 0
        self.error_message = None
        self.checksum_source = None
        self.checksum_destination = None
        self.estimated_duration_seconds = 0
        self.actual_duration_seconds = 0

    def calculate_estimated_duration(self) -> float:
        """Calculate estimated migration duration"""
        dest_config = CLOUD_CONFIGS[self.destination]
        size_mb = self.data_obj.size_mb

        # Transfer time based on bandwidth
        transfer_time_seconds = (size_mb * 8) / dest_config.transfer_speed_mbps

        # Add overhead for encryption/validation
        if self.security_level == SecurityLevel.MAXIMUM:
            transfer_time_seconds *= 1.3  # 30% overhead
        elif self.security_level == SecurityLevel.HIGH:
            transfer_time_seconds *= 1.15  # 15% overhead

        self.estimated_duration_seconds = transfer_time_seconds
        return transfer_time_seconds

    def calculate_cost(self) -> float:
        """Calculate migration cost"""
        dest_config = CLOUD_CONFIGS[self.destination]
        size_gb = self.data_obj.size_mb / 1024
        return size_gb * dest_config.bandwidth_cost_per_gb

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export"""
        return {
            'job_id': self.job_id,
            'file_id': self.data_obj.file_id,
            'file_name': self.data_obj.name,
            'size_mb': self.data_obj.size_mb,
            'source': self.source.value,
            'destination': self.destination.value,
            'priority': self.priority.value,
            'security_level': self.security_level.value,
            'status': self.status.value,
            'progress_percent': round(self.progress_percent, 2),
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'estimated_duration_seconds': round(self.estimated_duration_seconds, 2),
            'actual_duration_seconds': round(self.actual_duration_seconds, 2),
            'cost': round(self.calculate_cost(), 2),
            'error_message': self.error_message
        }


# ===================== SECURITY MANAGER =====================

class SecurityManager:
    """Handles security for migrations"""

    @staticmethod
    def generate_checksum(data_obj: DataObject) -> str:
        """Generate checksum for data integrity verification"""
        # Simulate checksum based on file properties
        data_string = f"{data_obj.file_id}{data_obj.name}{data_obj.size_mb}"
        return hashlib.sha256(data_string.encode()).hexdigest()

    @staticmethod
    def verify_checksum(source_checksum: str, dest_checksum: str) -> bool:
        """Verify checksums match"""
        return source_checksum == dest_checksum

    @staticmethod
    def encrypt_transfer(data_obj: DataObject, security_level: SecurityLevel) -> Dict:
        """Simulate encryption (returns metadata)"""
        return {
            'encrypted': True,
            'algorithm': 'AES-256' if security_level != SecurityLevel.STANDARD else 'AES-128',
            'key_id': f"key_{data_obj.file_id}",
            'timestamp': datetime.now().isoformat()
        }

    @staticmethod
    def validate_cloud_credentials(location: StorageLocation) -> bool:
        """Validate cloud provider credentials (simulated)"""
        # In real implementation, check API keys, tokens, etc.
        return True


# ===================== MIGRATION SCHEDULER =====================

class MigrationScheduler:
    """
    Schedules migrations to minimize disruption:
    - Prioritizes critical jobs
    - Throttles bandwidth usage
    - Schedules large transfers during off-peak hours
    """

    def __init__(self, max_concurrent_jobs: int = 3,
                 max_bandwidth_mbps: float = 1000):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_bandwidth_mbps = max_bandwidth_mbps
        self.pending_queue = []
        self.active_jobs = []
        self.completed_jobs = []
        self.failed_jobs = []

    def add_job(self, job: MigrationJob):
        """Add job to scheduler"""
        job.calculate_estimated_duration()
        self.pending_queue.append(job)
        # Sort by priority
        self.pending_queue.sort(key=lambda j: j.priority.value)

    def get_next_jobs(self) -> List[MigrationJob]:
        """Get next jobs to execute based on priority and resources"""
        available_slots = self.max_concurrent_jobs - len(self.active_jobs)

        if available_slots <= 0:
            return []

        # Get highest priority jobs that fit bandwidth
        next_jobs = []
        total_bandwidth = sum(
            CLOUD_CONFIGS[j.destination].transfer_speed_mbps
            for j in self.active_jobs
        )

        for job in self.pending_queue[:]:
            if len(next_jobs) >= available_slots:
                break

            job_bandwidth = CLOUD_CONFIGS[job.destination].transfer_speed_mbps

            if total_bandwidth + job_bandwidth <= self.max_bandwidth_mbps:
                next_jobs.append(job)
                total_bandwidth += job_bandwidth
                self.pending_queue.remove(job)

        return next_jobs

    def should_defer_to_offpeak(self, job: MigrationJob) -> bool:
        """Determine if large job should wait for off-peak hours"""
        # Defer large, low-priority jobs to off-peak (simulated)
        if job.priority == MigrationPriority.LOW and job.data_obj.size_mb > 1000:
            current_hour = datetime.now().hour
            # Off-peak: 10 PM to 6 AM
            if not (22 <= current_hour or current_hour <= 6):
                return True
        return False

    def get_statistics(self) -> Dict:
        """Get scheduler statistics"""
        return {
            'pending': len(self.pending_queue),
            'active': len(self.active_jobs),
            'completed': len(self.completed_jobs),
            'failed': len(self.failed_jobs),
            'total': len(self.pending_queue) + len(self.active_jobs) +
                    len(self.completed_jobs) + len(self.failed_jobs)
        }


# ===================== SYNCHRONIZATION ENGINE =====================

class SyncEngine:
    """
    Handles data synchronization across clouds:
    - Keeps replicas in sync
    - Detects conflicts
    - Ensures consistency
    """

    def __init__(self):
        self.sync_jobs = []
        self.replicas = {}  # file_id -> [locations]

    def create_replica(self, data_obj: DataObject, locations: List[StorageLocation]):
        """Create replicas across multiple clouds"""
        self.replicas[data_obj.file_id] = {
            'primary': data_obj.location,
            'replicas': locations,
            'last_sync': datetime.now(),
            'checksum': SecurityManager.generate_checksum(data_obj),
            'status': 'synced'
        }

    def detect_conflicts(self, file_id: str) -> Optional[Dict]:
        """Detect if replicas are out of sync"""
        if file_id not in self.replicas:
            return None

        # Simulate conflict detection
        replica_info = self.replicas[file_id]
        time_since_sync = (datetime.now() - replica_info['last_sync']).total_seconds()

        # If not synced in 1 hour, might have conflict
        if time_since_sync > 3600:
            return {
                'file_id': file_id,
                'conflict_type': 'stale_replica',
                'time_since_sync': time_since_sync,
                'resolution': 'sync_from_primary'
            }

        return None

    def sync_replicas(self, file_id: str) -> List[MigrationJob]:
        """Create sync jobs to update replicas"""
        if file_id not in self.replicas:
            return []

        replica_info = self.replicas[file_id]
        # Would create migration jobs to sync replicas
        # For now, just update last_sync
        replica_info['last_sync'] = datetime.now()
        replica_info['status'] = 'synced'

        return []


# ===================== MIGRATION ENGINE =====================

class MultiCloudMigrationEngine:
    """Main engine for multi-cloud migrations"""

    def __init__(self):
        self.scheduler = MigrationScheduler(max_concurrent_jobs=3, max_bandwidth_mbps=1000)
        self.security_manager = SecurityManager()
        self.sync_engine = SyncEngine()
        self.job_counter = 0

    def create_migration_job(self, data_obj: DataObject,
                            destination: StorageLocation,
                            priority: MigrationPriority = MigrationPriority.NORMAL,
                            security_level: SecurityLevel = SecurityLevel.HIGH) -> MigrationJob:
        """Create a new migration job"""
        self.job_counter += 1
        job_id = f"MIG_{self.job_counter:05d}"

        job = MigrationJob(
            job_id=job_id,
            data_obj=data_obj,
            source=data_obj.location,
            destination=destination,
            priority=priority,
            security_level=security_level
        )

        self.scheduler.add_job(job)
        return job

    def execute_migration(self, job: MigrationJob, simulate: bool = True) -> bool:
        """
        Execute a migration with security and performance optimization
        """
        # 1. Validate credentials
        if not self.security_manager.validate_cloud_credentials(job.destination):
            job.status = MigrationStatus.FAILED
            job.error_message = "Invalid credentials"
            return False

        # 2. Generate source checksum
        job.checksum_source = self.security_manager.generate_checksum(job.data_obj)

        # 3. Encrypt for transfer
        encryption_info = self.security_manager.encrypt_transfer(
            job.data_obj, job.security_level
        )

        # 4. Start migration
        job.status = MigrationStatus.IN_PROGRESS
        job.started_at = datetime.now()

        if simulate:
            # Simulate transfer
            duration = job.estimated_duration_seconds
            steps = 10
            for i in range(steps + 1):
                job.progress_percent = (i / steps) * 100
                job.bytes_transferred = (job.data_obj.size_mb * 1024 * 1024) * (i / steps)
                if not simulate:
                    time.sleep(duration / steps)

        # 5. Verify checksum
        job.checksum_destination = job.checksum_source  # Would verify from destination

        if not self.security_manager.verify_checksum(
            job.checksum_source, job.checksum_destination
        ):
            job.status = MigrationStatus.FAILED
            job.error_message = "Checksum mismatch"
            return False

        # 6. Complete
        job.status = MigrationStatus.COMPLETED
        job.completed_at = datetime.now()
        job.actual_duration_seconds = (job.completed_at - job.started_at).total_seconds()
        job.progress_percent = 100.0

        # Update data object location
        job.data_obj.location = job.destination

        return True

    def execute_all_pending(self, simulate: bool = True) -> Dict:
        """Execute all pending migrations"""
        results = {
            'total_jobs': 0,
            'successful': 0,
            'failed': 0,
            'total_cost': 0.0,
            'total_data_transferred_gb': 0.0,
            'jobs': []
        }

        while self.scheduler.pending_queue:
            next_jobs = self.scheduler.get_next_jobs()

            if not next_jobs:
                break

            for job in next_jobs:
                results['total_jobs'] += 1
                self.scheduler.active_jobs.append(job)

                success = self.execute_migration(job, simulate)

                if success:
                    results['successful'] += 1
                    results['total_cost'] += job.calculate_cost()
                    results['total_data_transferred_gb'] += job.data_obj.size_mb / 1024
                    self.scheduler.completed_jobs.append(job)
                else:
                    results['failed'] += 1
                    self.scheduler.failed_jobs.append(job)

                self.scheduler.active_jobs.remove(job)
                results['jobs'].append(job.to_dict())

        return results

    def get_migration_status(self) -> Dict:
        """Get current migration status for dashboard"""
        return {
            'timestamp': datetime.now().isoformat(),
            'scheduler': self.scheduler.get_statistics(),
            'active_jobs': [j.to_dict() for j in self.scheduler.active_jobs],
            'pending_jobs': [j.to_dict() for j in self.scheduler.pending_queue[:10]],  # Top 10
            'recent_completed': [j.to_dict() for j in self.scheduler.completed_jobs[-10:]],  # Last 10
            'failed_jobs': [j.to_dict() for j in self.scheduler.failed_jobs]
        }

    def export_to_json(self, filename: str = "migration_status.json"):
        """Export migration data for dashboard"""
        data = self.get_migration_status()
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Migration status exported to {filename}")


# ===================== EXPORT FUNCTIONS =====================

def export_migration_report(results: Dict, filename: str = "migration_report.json"):
    """Export migration results"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Migration report exported to {filename}")


if __name__ == "__main__":
    print("Task 2: Multi-Cloud Migration Engine - Ready!")
    print("Import this module to use migration features")

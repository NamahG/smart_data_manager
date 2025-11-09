"""
Task 5: Data Consistency and Availability
Ensures data consistency across distributed environments and handles network failures gracefully
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json
import hashlib
import random
from component1_data_sorter import DataObject, StorageLocation


# ===================== ENUMS =====================

class ConflictType(Enum):
    """Types of data conflicts"""
    VERSION_CONFLICT = "version_conflict"  # Different versions exist
    CHECKSUM_MISMATCH = "checksum_mismatch"  # Data corruption
    STALE_REPLICA = "stale_replica"  # Replica not updated
    NETWORK_PARTITION = "network_partition"  # Split-brain scenario


class ResolutionStrategy(Enum):
    """Conflict resolution strategies"""
    LAST_WRITE_WINS = "last_write_wins"  # Newest version wins
    MANUAL_REVIEW = "manual_review"  # Human decision needed
    MERGE = "merge"  # Attempt automatic merge
    RESTORE_PRIMARY = "restore_primary"  # Restore from primary source


class HealthStatus(Enum):
    """Health status of storage nodes"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNREACHABLE = "unreachable"
    FAILED = "failed"


# ===================== DATA VERSION =====================

class DataVersion:
    """Represents a version of a data object"""

    def __init__(self, file_id: str, version: int, checksum: str,
                 location: StorageLocation, modified_at: datetime):
        self.file_id = file_id
        self.version = version
        self.checksum = checksum
        self.location = location
        self.modified_at = modified_at
        self.is_primary = False

    def to_dict(self) -> Dict:
        return {
            'file_id': self.file_id,
            'version': self.version,
            'checksum': self.checksum,
            'location': self.location.value,
            'modified_at': self.modified_at.isoformat(),
            'is_primary': self.is_primary
        }


# ===================== CONFLICT =====================

class DataConflict:
    """Represents a detected conflict"""

    def __init__(self, file_id: str, conflict_type: ConflictType,
                 versions: List[DataVersion], detected_at: datetime):
        self.conflict_id = f"CONFLICT_{int(datetime.now().timestamp() * 1000)}"
        self.file_id = file_id
        self.conflict_type = conflict_type
        self.versions = versions
        self.detected_at = detected_at
        self.resolved = False
        self.resolution_strategy = None
        self.resolved_at = None

    def to_dict(self) -> Dict:
        return {
            'conflict_id': self.conflict_id,
            'file_id': self.file_id,
            'conflict_type': self.conflict_type.value,
            'versions_count': len(self.versions),
            'versions': [v.to_dict() for v in self.versions],
            'detected_at': self.detected_at.isoformat(),
            'resolved': self.resolved,
            'resolution_strategy': self.resolution_strategy.value if self.resolution_strategy else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


# ===================== STORAGE NODE =====================

class StorageNode:
    """Represents a storage location with health monitoring"""

    def __init__(self, location: StorageLocation):
        self.location = location
        self.status = HealthStatus.HEALTHY
        self.last_check = datetime.now()
        self.uptime_percent = 100.0
        self.failures = 0
        self.network_latency_ms = 0

    def simulate_health_check(self) -> bool:
        """Simulate health check (with random failures for demo)"""
        self.last_check = datetime.now()

        # Simulate occasional failures
        if random.random() < 0.05:  # 5% chance of failure
            self.status = HealthStatus.UNREACHABLE
            self.failures += 1
            self.uptime_percent = max(0, self.uptime_percent - 5)
            return False

        self.status = HealthStatus.HEALTHY
        self.network_latency_ms = random.randint(5, 100)
        self.uptime_percent = min(100, self.uptime_percent + 1)
        return True

    def to_dict(self) -> Dict:
        return {
            'location': self.location.value,
            'status': self.status.value,
            'uptime_percent': round(self.uptime_percent, 2),
            'failures': self.failures,
            'network_latency_ms': self.network_latency_ms,
            'last_check': self.last_check.isoformat()
        }


# ===================== VERSION MANAGER =====================

class VersionManager:
    """Manages versions of data objects across locations"""

    def __init__(self):
        self.versions = {}  # file_id -> List[DataVersion]
        self.version_counter = {}  # file_id -> current_version

    def create_version(self, data_obj: DataObject) -> DataVersion:
        """Create new version for data object"""
        file_id = data_obj.file_id

        # Get next version number
        if file_id not in self.version_counter:
            self.version_counter[file_id] = 1
        else:
            self.version_counter[file_id] += 1

        # Generate checksum
        checksum = self._generate_checksum(data_obj)

        # Create version
        version = DataVersion(
            file_id=file_id,
            version=self.version_counter[file_id],
            checksum=checksum,
            location=data_obj.location,
            modified_at=datetime.now()
        )

        # Store
        if file_id not in self.versions:
            self.versions[file_id] = []
            version.is_primary = True

        self.versions[file_id].append(version)

        return version

    def _generate_checksum(self, data_obj: DataObject) -> str:
        """Generate checksum for data object"""
        data_str = f"{data_obj.file_id}{data_obj.name}{data_obj.size_mb}{datetime.now().timestamp()}"
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def get_latest_version(self, file_id: str) -> Optional[DataVersion]:
        """Get latest version for file"""
        if file_id not in self.versions or not self.versions[file_id]:
            return None

        return max(self.versions[file_id], key=lambda v: v.version)

    def get_versions_at_location(self, file_id: str, location: StorageLocation) -> List[DataVersion]:
        """Get all versions at specific location"""
        if file_id not in self.versions:
            return []

        return [v for v in self.versions[file_id] if v.location == location]


# ===================== CONFLICT DETECTOR =====================

class ConflictDetector:
    """Detects conflicts across distributed copies"""

    def __init__(self, version_manager: VersionManager):
        self.version_manager = version_manager
        self.detected_conflicts = []

    def detect_conflicts(self, file_id: str) -> List[DataConflict]:
        """Detect conflicts for a file across locations"""
        conflicts = []

        versions = self.version_manager.versions.get(file_id, [])
        if len(versions) < 2:
            return conflicts

        # Group by location
        location_versions = {}
        for v in versions:
            loc = v.location.value
            if loc not in location_versions:
                location_versions[loc] = []
            location_versions[loc].append(v)

        # Check for version conflicts
        if len(location_versions) > 1:
            latest_versions = []
            for loc, vers in location_versions.items():
                latest = max(vers, key=lambda v: v.version)
                latest_versions.append(latest)

            # If different versions exist, it's a conflict
            version_nums = [v.version for v in latest_versions]
            if len(set(version_nums)) > 1:
                conflict = DataConflict(
                    file_id=file_id,
                    conflict_type=ConflictType.VERSION_CONFLICT,
                    versions=latest_versions,
                    detected_at=datetime.now()
                )
                conflicts.append(conflict)
                self.detected_conflicts.append(conflict)

        # Check for checksum mismatches
        checksums = [v.checksum for v in versions if v.version == max(v.version for v in versions)]
        if len(set(checksums)) > 1:
            conflict = DataConflict(
                file_id=file_id,
                conflict_type=ConflictType.CHECKSUM_MISMATCH,
                versions=[v for v in versions if v.version == max(v.version for v in versions)],
                detected_at=datetime.now()
            )
            conflicts.append(conflict)
            self.detected_conflicts.append(conflict)

        return conflicts

    def detect_stale_replicas(self, file_id: str, max_age_hours: int = 24) -> Optional[DataConflict]:
        """Detect replicas that haven't been updated recently"""
        versions = self.version_manager.versions.get(file_id, [])
        if not versions:
            return None

        latest = self.version_manager.get_latest_version(file_id)
        stale_versions = []

        for v in versions:
            age = (datetime.now() - v.modified_at).total_seconds() / 3600
            if age > max_age_hours and v.version < latest.version:
                stale_versions.append(v)

        if stale_versions:
            conflict = DataConflict(
                file_id=file_id,
                conflict_type=ConflictType.STALE_REPLICA,
                versions=stale_versions,
                detected_at=datetime.now()
            )
            self.detected_conflicts.append(conflict)
            return conflict

        return None


# ===================== CONFLICT RESOLVER =====================

class ConflictResolver:
    """Resolves conflicts automatically or flags for manual review"""

    def __init__(self, version_manager: VersionManager):
        self.version_manager = version_manager
        self.resolved_conflicts = []

    def resolve(self, conflict: DataConflict, strategy: ResolutionStrategy = ResolutionStrategy.LAST_WRITE_WINS) -> bool:
        """Resolve a conflict using specified strategy"""

        if conflict.conflict_type == ConflictType.VERSION_CONFLICT:
            return self._resolve_version_conflict(conflict, strategy)
        elif conflict.conflict_type == ConflictType.CHECKSUM_MISMATCH:
            return self._resolve_checksum_conflict(conflict, strategy)
        elif conflict.conflict_type == ConflictType.STALE_REPLICA:
            return self._resolve_stale_replica(conflict)
        else:
            # Network partition - requires manual review
            conflict.resolution_strategy = ResolutionStrategy.MANUAL_REVIEW
            return False

    def _resolve_version_conflict(self, conflict: DataConflict, strategy: ResolutionStrategy) -> bool:
        """Resolve version conflict"""

        if strategy == ResolutionStrategy.LAST_WRITE_WINS:
            # Choose version with latest modification time
            winner = max(conflict.versions, key=lambda v: v.modified_at)

            # Mark as resolved
            conflict.resolved = True
            conflict.resolution_strategy = strategy
            conflict.resolved_at = datetime.now()
            self.resolved_conflicts.append(conflict)

            return True

        elif strategy == ResolutionStrategy.MANUAL_REVIEW:
            conflict.resolution_strategy = strategy
            return False

        return False

    def _resolve_checksum_conflict(self, conflict: DataConflict, strategy: ResolutionStrategy) -> bool:
        """Resolve checksum mismatch - usually requires re-download"""

        if strategy == ResolutionStrategy.RESTORE_PRIMARY:
            # Restore from primary version
            primary = next((v for v in conflict.versions if v.is_primary), None)

            if primary:
                conflict.resolved = True
                conflict.resolution_strategy = strategy
                conflict.resolved_at = datetime.now()
                self.resolved_conflicts.append(conflict)
                return True

        return False

    def _resolve_stale_replica(self, conflict: DataConflict) -> bool:
        """Resolve stale replica by syncing"""
        # Automatically sync to latest version
        conflict.resolved = True
        conflict.resolution_strategy = ResolutionStrategy.RESTORE_PRIMARY
        conflict.resolved_at = datetime.now()
        self.resolved_conflicts.append(conflict)
        return True


# ===================== HEALTH MONITOR =====================

class HealthMonitor:
    """Monitors health of all storage nodes"""

    def __init__(self):
        self.nodes = {}  # location -> StorageNode
        self.check_interval_seconds = 30
        self.last_full_check = datetime.now()

    def add_node(self, location: StorageLocation):
        """Add node to monitoring"""
        if location not in self.nodes:
            self.nodes[location] = StorageNode(location)

    def check_all_nodes(self) -> Dict[str, HealthStatus]:
        """Check health of all nodes"""
        results = {}

        for location, node in self.nodes.items():
            is_healthy = node.simulate_health_check()
            results[location.value] = node.status

        self.last_full_check = datetime.now()
        return results

    def get_healthy_nodes(self) -> List[StorageLocation]:
        """Get list of healthy nodes"""
        return [loc for loc, node in self.nodes.items()
                if node.status == HealthStatus.HEALTHY]

    def get_statistics(self) -> Dict:
        """Get health statistics"""
        total = len(self.nodes)
        healthy = len([n for n in self.nodes.values() if n.status == HealthStatus.HEALTHY])
        degraded = len([n for n in self.nodes.values() if n.status == HealthStatus.DEGRADED])
        failed = len([n for n in self.nodes.values() if n.status == HealthStatus.UNREACHABLE])

        avg_uptime = sum(n.uptime_percent for n in self.nodes.values()) / total if total > 0 else 0
        total_failures = sum(n.failures for n in self.nodes.values())

        return {
            'total_nodes': total,
            'healthy': healthy,
            'degraded': degraded,
            'failed': failed,
            'average_uptime_percent': round(avg_uptime, 2),
            'total_failures': total_failures,
            'last_check': self.last_full_check.isoformat()
        }


# ===================== CONSISTENCY MANAGER =====================

class ConsistencyManager:
    """Main manager for consistency and availability"""

    def __init__(self):
        self.version_manager = VersionManager()
        self.conflict_detector = ConflictDetector(self.version_manager)
        self.conflict_resolver = ConflictResolver(self.version_manager)
        self.health_monitor = HealthMonitor()

        # Initialize nodes for all storage locations
        for location in StorageLocation:
            self.health_monitor.add_node(location)

    def register_data_object(self, data_obj: DataObject) -> DataVersion:
        """Register a data object and create initial version"""
        return self.version_manager.create_version(data_obj)

    def update_data_object(self, data_obj: DataObject) -> DataVersion:
        """Update data object (creates new version)"""
        return self.version_manager.create_version(data_obj)

    def run_consistency_check(self, file_ids: List[str]) -> Dict:
        """Run full consistency check"""
        all_conflicts = []

        for file_id in file_ids:
            conflicts = self.conflict_detector.detect_conflicts(file_id)
            all_conflicts.extend(conflicts)

            # Also check for stale replicas
            stale = self.conflict_detector.detect_stale_replicas(file_id)
            if stale:
                all_conflicts.append(stale)

        return {
            'conflicts_detected': len(all_conflicts),
            'conflicts': [c.to_dict() for c in all_conflicts]
        }

    def auto_resolve_conflicts(self) -> Dict:
        """Automatically resolve conflicts where possible"""
        unresolved = [c for c in self.conflict_detector.detected_conflicts if not c.resolved]

        resolved_count = 0
        manual_review_count = 0

        for conflict in unresolved:
            if conflict.conflict_type == ConflictType.STALE_REPLICA:
                # Auto-resolve stale replicas
                success = self.conflict_resolver.resolve(conflict)
                if success:
                    resolved_count += 1
            elif conflict.conflict_type == ConflictType.VERSION_CONFLICT:
                # Use last-write-wins strategy
                success = self.conflict_resolver.resolve(conflict, ResolutionStrategy.LAST_WRITE_WINS)
                if success:
                    resolved_count += 1
            else:
                # Requires manual review
                conflict.resolution_strategy = ResolutionStrategy.MANUAL_REVIEW
                manual_review_count += 1

        return {
            'total_conflicts': len(unresolved),
            'auto_resolved': resolved_count,
            'manual_review_needed': manual_review_count
        }

    def check_system_health(self) -> Dict:
        """Check health of entire system"""
        node_health = self.health_monitor.check_all_nodes()
        stats = self.health_monitor.get_statistics()

        return {
            'node_health': node_health,
            'statistics': stats,
            'healthy_nodes': [loc.value for loc in self.health_monitor.get_healthy_nodes()]
        }

    def export_status(self, filename: str = "consistency_status.json"):
        """Export consistency status for dashboard"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'health': self.health_monitor.get_statistics(),
            'conflicts': {
                'total_detected': len(self.conflict_detector.detected_conflicts),
                'resolved': len(self.conflict_resolver.resolved_conflicts),
                'unresolved': len([c for c in self.conflict_detector.detected_conflicts if not c.resolved])
            },
            'nodes': [node.to_dict() for node in self.health_monitor.nodes.values()]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Consistency status exported to {filename}")


if __name__ == "__main__":
    print("Task 5: Data Consistency and Availability - Ready!")
    print("Import this module to use consistency features")

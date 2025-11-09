"""
Component 1: Automatic Data Sorter (Smart Data Placement System)
A prototype for intelligent data classification and placement across storage tiers

Student-friendly implementation for hackathon project
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional
import json


# ===================== DATA MODELS =====================

class StorageTier(Enum):
    """Storage tiers with different performance and cost characteristics"""
    HOT = "hot"      # Fast, expensive (e.g., SSD, Premium storage)
    WARM = "warm"    # Medium speed, medium cost (Standard storage)
    COLD = "cold"    # Slow, cheap (Archive storage like AWS Glacier)


class StorageLocation(Enum):
    """Different storage environments"""
    ON_PREMISE = "on-premise"
    PRIVATE_CLOUD = "private-cloud"
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class LatencyRequirement(Enum):
    """Latency requirements for applications"""
    CRITICAL = "critical"    # <10ms required (real-time apps, trading, gaming)
    STANDARD = "standard"    # <100ms acceptable (web apps, APIs)
    FLEXIBLE = "flexible"    # >1s acceptable (batch processing, archives)


class DataObject:
    """Represents a file or data object in the system"""

    def __init__(self, file_id: str, name: str, size_mb: float,
                 location: StorageLocation, tier: StorageTier,
                 latency_requirement: LatencyRequirement = LatencyRequirement.STANDARD):
        self.file_id = file_id
        self.name = name
        self.size_mb = size_mb
        self.location = location
        self.tier = tier
        self.latency_requirement = latency_requirement
        self.access_count = 0
        self.last_accessed = datetime.now()
        self.created_at = datetime.now()
        self.access_history = []  # List of access timestamps

    def access(self):
        """Simulate accessing this file"""
        self.access_count += 1
        self.last_accessed = datetime.now()
        self.access_history.append(datetime.now())

    def get_access_frequency(self, days: int = 30) -> float:
        """Calculate access frequency (accesses per day) over recent period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_accesses = [a for a in self.access_history if a > cutoff_date]
        return len(recent_accesses) / days if days > 0 else 0

    def days_since_last_access(self) -> int:
        """Days since last access"""
        return (datetime.now() - self.last_accessed).days

    def get_access_trend(self) -> str:
        """
        Analyze access trend to predict future behavior
        Returns: 'increasing', 'decreasing', 'stable', or 'insufficient_data'
        """
        if len(self.access_history) < 6:
            return 'insufficient_data'

        # Compare recent vs older access patterns
        recent_freq = self.get_access_frequency(days=7)  # Last week
        older_freq = self.get_access_frequency(days=30)  # Last month average

        if older_freq == 0:
            return 'insufficient_data'

        change_ratio = recent_freq / older_freq

        if change_ratio > 1.5:
            return 'increasing'
        elif change_ratio < 0.5:
            return 'decreasing'
        else:
            return 'stable'

    def predict_future_tier(self) -> Optional[StorageTier]:
        """
        Predict what tier this data will need in the future based on trends
        This is basic trend analysis - full ML prediction is in Task 4
        """
        trend = self.get_access_trend()
        current_freq = self.get_access_frequency(days=30)

        if trend == 'increasing':
            # Access is increasing - might need faster storage
            if current_freq > 0.5:
                return StorageTier.HOT
            elif current_freq > 0.1:
                return StorageTier.WARM
        elif trend == 'decreasing':
            # Access is decreasing - can move to cheaper storage
            if current_freq < 0.1:
                return StorageTier.COLD
            elif current_freq < 0.5:
                return StorageTier.WARM

        # If stable or insufficient data, return None (use current classification)
        return None

    def __repr__(self):
        return (f"DataObject(id={self.file_id}, name={self.name}, "
                f"tier={self.tier.value}, location={self.location.value})")


# ===================== STORAGE TIER CONFIGURATION =====================

class TierConfig:
    """Configuration for each storage tier"""

    def __init__(self, tier: StorageTier, cost_per_gb_month: float,
                 latency_ms: int, transfer_cost_per_gb: float):
        self.tier = tier
        self.cost_per_gb_month = cost_per_gb_month
        self.latency_ms = latency_ms
        self.transfer_cost_per_gb = transfer_cost_per_gb


# Realistic cost and performance models
TIER_CONFIGS = {
    StorageTier.HOT: TierConfig(
        tier=StorageTier.HOT,
        cost_per_gb_month=0.15,  # $0.15/GB/month (expensive but fast)
        latency_ms=5,             # 5ms latency
        transfer_cost_per_gb=0.01
    ),
    StorageTier.WARM: TierConfig(
        tier=StorageTier.WARM,
        cost_per_gb_month=0.05,  # $0.05/GB/month (medium)
        latency_ms=50,            # 50ms latency
        transfer_cost_per_gb=0.005
    ),
    StorageTier.COLD: TierConfig(
        tier=StorageTier.COLD,
        cost_per_gb_month=0.01,  # $0.01/GB/month (cheap but slow)
        latency_ms=5000,          # 5 seconds latency
        transfer_cost_per_gb=0.05 # Higher retrieval cost
    )
}


# ===================== DATA CLASSIFIER =====================

class DataClassifier:
    """
    Classifies data into HOT, WARM, or COLD based on:
    1. Access frequency
    2. Latency requirements
    3. Cost considerations
    4. Predicted future trends
    """

    # Classification thresholds (customizable)
    HOT_THRESHOLD_ACCESSES_PER_DAY = 1.0    # >1 access/day = HOT
    WARM_THRESHOLD_DAYS_SINCE_ACCESS = 30   # Not accessed in 30 days = COLD
    WARM_MIN_ACCESSES_PER_DAY = 0.1         # <0.1 access/day but recent = WARM

    @classmethod
    def classify(cls, data_obj: DataObject, consider_trends: bool = True,
                 consider_latency: bool = True) -> StorageTier:
        """
        Classify data based on ALL FOUR Task 1 factors:
        1. Access frequency (hot, warm, cold data)
        2. Latency requirements
        3. Cost per GB (implicit in tier selection)
        4. Predicted future access trends

        Args:
            data_obj: The data object to classify
            consider_trends: Whether to factor in access trends
            consider_latency: Whether to factor in latency requirements
        """
        access_freq = data_obj.get_access_frequency(days=30)
        days_since_access = data_obj.days_since_last_access()

        # FACTOR 1: Access Frequency - Base classification
        if access_freq >= cls.HOT_THRESHOLD_ACCESSES_PER_DAY:
            base_tier = StorageTier.HOT
        elif days_since_access >= cls.WARM_THRESHOLD_DAYS_SINCE_ACCESS:
            base_tier = StorageTier.COLD
        else:
            base_tier = StorageTier.WARM

        # FACTOR 2: Latency Requirements - Override if critical
        if consider_latency:
            if data_obj.latency_requirement == LatencyRequirement.CRITICAL:
                # Critical latency apps MUST be on HOT tier
                if base_tier != StorageTier.HOT:
                    return StorageTier.HOT  # Override for latency
            elif data_obj.latency_requirement == LatencyRequirement.FLEXIBLE:
                # Flexible latency can use cheaper tiers even if accessed somewhat
                if base_tier == StorageTier.HOT and access_freq < 2.0:
                    base_tier = StorageTier.WARM  # Downgrade to save cost

        # FACTOR 3: Cost is implicit (cheaper tier preferred when possible)
        # FACTOR 4: Predicted Future Trends - Adjust based on trend
        if consider_trends:
            trend = data_obj.get_access_trend()
            predicted_tier = data_obj.predict_future_tier()

            if trend == 'increasing' and predicted_tier:
                # If access is increasing, proactively move to faster tier
                if predicted_tier.value == 'hot' and base_tier != StorageTier.HOT:
                    return StorageTier.HOT
                elif predicted_tier.value == 'warm' and base_tier == StorageTier.COLD:
                    return StorageTier.WARM

            elif trend == 'decreasing' and predicted_tier:
                # If access is decreasing, can move to cheaper tier
                if predicted_tier.value == 'cold' and base_tier != StorageTier.COLD:
                    # But respect latency requirements
                    if data_obj.latency_requirement != LatencyRequirement.CRITICAL:
                        return StorageTier.COLD

        return base_tier

    @classmethod
    def get_classification_reason(cls, data_obj: DataObject) -> str:
        """Get human-readable reason for classification with ALL 4 factors"""
        access_freq = data_obj.get_access_frequency(days=30)
        days_since_access = data_obj.days_since_last_access()
        trend = data_obj.get_access_trend()
        latency_req = data_obj.latency_requirement.value

        reasons = []

        # Factor 1: Access frequency
        if access_freq >= cls.HOT_THRESHOLD_ACCESSES_PER_DAY:
            reasons.append(f"Freq: {access_freq:.2f}/day")
        elif days_since_access >= cls.WARM_THRESHOLD_DAYS_SINCE_ACCESS:
            reasons.append(f"Last access: {days_since_access}d ago")
        else:
            reasons.append(f"Freq: {access_freq:.2f}/day")

        # Factor 2: Latency requirement
        if latency_req == 'critical':
            reasons.append("Latency: CRITICAL")
        elif latency_req == 'flexible':
            reasons.append("Latency: flexible")

        # Factor 4: Trend
        if trend == 'increasing':
            reasons.append("Trend: ↑ increasing")
        elif trend == 'decreasing':
            reasons.append("Trend: ↓ decreasing")
        elif trend == 'stable':
            reasons.append("Trend: → stable")

        # Factor 3: Cost is implicit but mentioned
        reasons.append("Cost-optimized")

        return ", ".join(reasons)


# ===================== PLACEMENT OPTIMIZER =====================

class PlacementOptimizer:
    """Optimizes data placement to minimize costs while meeting performance needs"""

    def __init__(self):
        self.classifier = DataClassifier()

    def analyze_data(self, data_obj: DataObject) -> Dict:
        """
        Analyze a data object and recommend optimal placement
        Considers ALL 4 Task 1 factors:
        1. Access frequency
        2. Latency requirements
        3. Cost per GB
        4. Predicted future trends
        """
        recommended_tier = self.classifier.classify(data_obj)
        current_tier = data_obj.tier

        needs_migration = recommended_tier != current_tier

        # Calculate costs (Factor 3)
        current_cost = self._calculate_monthly_cost(data_obj, current_tier)
        recommended_cost = self._calculate_monthly_cost(data_obj, recommended_tier)
        savings = current_cost - recommended_cost

        # Get latency info (Factor 2)
        current_latency = TIER_CONFIGS[current_tier].latency_ms
        recommended_latency = TIER_CONFIGS[recommended_tier].latency_ms

        return {
            'file_id': data_obj.file_id,
            'file_name': data_obj.name,
            'current_tier': current_tier.value,
            'recommended_tier': recommended_tier.value,
            'needs_migration': needs_migration,
            'current_monthly_cost': round(current_cost, 2),
            'recommended_monthly_cost': round(recommended_cost, 2),
            'monthly_savings': round(savings, 2),
            'current_latency_ms': current_latency,
            'recommended_latency_ms': recommended_latency,
            'latency_requirement': data_obj.latency_requirement.value,
            'access_frequency': round(data_obj.get_access_frequency(days=30), 2),
            'access_trend': data_obj.get_access_trend(),
            'reason': self.classifier.get_classification_reason(data_obj)
        }

    def _calculate_monthly_cost(self, data_obj: DataObject, tier: StorageTier) -> float:
        """Calculate monthly storage cost for a data object"""
        config = TIER_CONFIGS[tier]
        size_gb = data_obj.size_mb / 1024
        return size_gb * config.cost_per_gb_month

    def optimize_all(self, data_objects: List[DataObject]) -> Dict:
        """Analyze all data objects and generate optimization report"""
        analyses = [self.analyze_data(obj) for obj in data_objects]

        migrations_needed = [a for a in analyses if a['needs_migration']]
        total_savings = sum(a['monthly_savings'] for a in migrations_needed)

        return {
            'total_files': len(data_objects),
            'files_needing_migration': len(migrations_needed),
            'total_monthly_savings': round(total_savings, 2),
            'analyses': analyses,
            'migrations': migrations_needed
        }


# ===================== DATA MIGRATION ENGINE =====================

class MigrationEngine:
    """Handles data migration between tiers"""

    def __init__(self):
        self.migration_history = []

    def migrate(self, data_obj: DataObject, target_tier: StorageTier) -> Dict:
        """Migrate a data object to a new tier"""
        old_tier = data_obj.tier

        if old_tier == target_tier:
            return {
                'success': False,
                'message': f'File already in {target_tier.value} tier'
            }

        # Calculate migration cost
        size_gb = data_obj.size_mb / 1024
        migration_cost = size_gb * TIER_CONFIGS[target_tier].transfer_cost_per_gb

        # Perform migration (simulated)
        data_obj.tier = target_tier

        migration_record = {
            'timestamp': datetime.now().isoformat(),
            'file_id': data_obj.file_id,
            'file_name': data_obj.name,
            'from_tier': old_tier.value,
            'to_tier': target_tier.value,
            'size_mb': data_obj.size_mb,
            'migration_cost': round(migration_cost, 2),
            'success': True
        }

        self.migration_history.append(migration_record)

        return migration_record

    def auto_migrate_all(self, data_objects: List[DataObject]) -> Dict:
        """Automatically migrate all files to their optimal tiers"""
        optimizer = PlacementOptimizer()
        migrations = []

        for data_obj in data_objects:
            analysis = optimizer.analyze_data(data_obj)

            if analysis['needs_migration']:
                recommended_tier = StorageTier(analysis['recommended_tier'])
                migration_result = self.migrate(data_obj, recommended_tier)
                migrations.append(migration_result)

        total_migration_cost = sum(m['migration_cost'] for m in migrations)

        return {
            'total_migrations': len(migrations),
            'total_migration_cost': round(total_migration_cost, 2),
            'migrations': migrations
        }


# ===================== SMART DATA MANAGER (Main System) =====================

class SmartDataManager:
    """Main system that ties everything together"""

    def __init__(self):
        self.data_objects: List[DataObject] = []
        self.optimizer = PlacementOptimizer()
        self.migrator = MigrationEngine()

    def add_data(self, data_obj: DataObject):
        """Add a data object to the system"""
        self.data_objects.append(data_obj)

    def run_optimization(self) -> Dict:
        """Run full optimization analysis"""
        return self.optimizer.optimize_all(self.data_objects)

    def auto_optimize(self) -> Dict:
        """Automatically optimize and migrate all data"""
        # First, analyze
        optimization = self.run_optimization()

        # Then, migrate
        migration_result = self.migrator.auto_migrate_all(self.data_objects)

        return {
            'optimization_analysis': optimization,
            'migration_results': migration_result
        }

    def get_statistics(self) -> Dict:
        """Get current system statistics"""
        tier_distribution = {
            'hot': 0,
            'warm': 0,
            'cold': 0
        }

        total_size_mb = 0
        total_cost = 0

        for obj in self.data_objects:
            tier_distribution[obj.tier.value] += 1
            total_size_mb += obj.size_mb

            size_gb = obj.size_mb / 1024
            config = TIER_CONFIGS[obj.tier]
            total_cost += size_gb * config.cost_per_gb_month

        return {
            'total_files': len(self.data_objects),
            'total_size_mb': round(total_size_mb, 2),
            'total_size_gb': round(total_size_mb / 1024, 2),
            'monthly_cost': round(total_cost, 2),
            'tier_distribution': tier_distribution
        }

    def print_report(self):
        """Print a human-readable report"""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print("SMART DATA MANAGER - SYSTEM REPORT")
        print("="*60)
        print(f"Total Files: {stats['total_files']}")
        print(f"Total Storage: {stats['total_size_gb']:.2f} GB")
        print(f"Monthly Cost: ${stats['monthly_cost']:.2f}")
        print(f"\nTier Distribution:")
        print(f"  HOT:  {stats['tier_distribution']['hot']} files")
        print(f"  WARM: {stats['tier_distribution']['warm']} files")
        print(f"  COLD: {stats['tier_distribution']['cold']} files")
        print("="*60 + "\n")


# ===================== EXPORT FUNCTIONS =====================

def export_report_json(report: Dict, filename: str = "optimization_report.json"):
    """Export optimization report to JSON"""
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report exported to {filename}")


if __name__ == "__main__":
    print("Component 1: Automatic Data Sorter - Ready!")
    print("Import this module in your demo script to use the Smart Data Manager")

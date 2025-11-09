"""
Component 6: Security, Encryption & Automated Alerts
Implements data encryption, access control policies, and threshold-based alerts
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
from enum import Enum
import json
import hashlib


class EncryptionLevel(Enum):
    """Encryption levels based on data sensitivity"""
    NONE = "none"
    STANDARD = "standard"          # AES-128
    ENHANCED = "enhanced"           # AES-256
    MILITARY_GRADE = "military"    # AES-256 + Multi-layer


class AccessPolicy(Enum):
    """Access control policies"""
    PUBLIC = "public"
    PRIVATE = "private"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"


class AlertType(Enum):
    """Types of alerts"""
    COST_THRESHOLD = "cost_threshold"
    LATENCY_THRESHOLD = "latency_threshold"
    SECURITY_VIOLATION = "security_violation"
    CAPACITY_WARNING = "capacity_warning"
    PERFORMANCE_DEGRADATION = "performance_degradation"


class SecurityPolicy:
    """Security policy for a file based on storage location"""

    def __init__(self, file_id: str, storage_location: str, data_classification: str):
        self.file_id = file_id
        self.storage_location = storage_location
        self.data_classification = data_classification
        self.encryption_level = self._determine_encryption_level()
        self.access_policy = self._determine_access_policy()
        self.encryption_key_id = self._generate_key_id()
        self.allowed_regions = self._determine_allowed_regions()

    def _determine_encryption_level(self) -> EncryptionLevel:
        """Adaptive encryption based on location and classification"""
        if self.data_classification == "confidential":
            return EncryptionLevel.MILITARY_GRADE
        elif self.data_classification == "sensitive":
            if self.storage_location in ["aws", "azure", "gcp"]:
                return EncryptionLevel.ENHANCED
            return EncryptionLevel.STANDARD
        elif self.storage_location in ["aws", "azure", "gcp"]:
            return EncryptionLevel.STANDARD
        return EncryptionLevel.NONE

    def _determine_access_policy(self) -> AccessPolicy:
        """Adaptive access control based on classification"""
        classification_map = {
            "confidential": AccessPolicy.CONFIDENTIAL,
            "sensitive": AccessPolicy.RESTRICTED,
            "internal": AccessPolicy.PRIVATE,
            "public": AccessPolicy.PUBLIC
        }
        return classification_map.get(self.data_classification, AccessPolicy.PRIVATE)

    def _generate_key_id(self) -> str:
        """Generate encryption key identifier"""
        if self.encryption_level == EncryptionLevel.NONE:
            return None
        hash_input = f"{self.file_id}_{self.storage_location}_{self.encryption_level.value}"
        return "KEY_" + hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _determine_allowed_regions(self) -> List[str]:
        """Determine allowed access regions based on policy"""
        if self.access_policy == AccessPolicy.CONFIDENTIAL:
            return ["on-premise"]
        elif self.access_policy == AccessPolicy.RESTRICTED:
            return ["on-premise", "private-cloud"]
        return ["on-premise", "private-cloud", "aws", "azure", "gcp"]

    def to_dict(self) -> Dict:
        return {
            'file_id': self.file_id,
            'storage_location': self.storage_location,
            'data_classification': self.data_classification,
            'encryption_level': self.encryption_level.value,
            'access_policy': self.access_policy.value,
            'encryption_key_id': self.encryption_key_id,
            'allowed_regions': self.allowed_regions,
            'encryption_at_rest': self.encryption_level != EncryptionLevel.NONE,
            'encryption_in_transit': True
        }


class Alert:
    """System alert based on threshold violations"""

    def __init__(self, alert_type: AlertType, severity: str, message: str,
                 current_value: float, threshold: float, resource_id: str = None):
        self.alert_id = f"ALERT_{int(datetime.now().timestamp() * 1000)}"
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.current_value = current_value
        self.threshold = threshold
        self.resource_id = resource_id
        self.timestamp = datetime.now().isoformat()
        self.acknowledged = False

    def to_dict(self) -> Dict:
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'severity': self.severity,
            'message': self.message,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'resource_id': self.resource_id,
            'timestamp': self.timestamp,
            'acknowledged': self.acknowledged
        }


class AlertingEngine:
    """Automated alerting based on configurable thresholds"""

    def __init__(self):
        self.cost_threshold_monthly = 100.0
        self.latency_threshold_ms = 500
        self.capacity_threshold_percent = 85
        self.error_rate_threshold_percent = 5.0
        self.alerts = []

    def check_cost_threshold(self, current_cost: float, resource_id: str) -> Optional[Alert]:
        """Check if cost exceeds threshold"""
        if current_cost > self.cost_threshold_monthly:
            alert = Alert(
                alert_type=AlertType.COST_THRESHOLD,
                severity="high" if current_cost > self.cost_threshold_monthly * 1.5 else "medium",
                message=f"Monthly cost ${current_cost:.2f} exceeds threshold ${self.cost_threshold_monthly:.2f}",
                current_value=current_cost,
                threshold=self.cost_threshold_monthly,
                resource_id=resource_id
            )
            self.alerts.append(alert)
            return alert
        return None

    def check_latency_threshold(self, current_latency_ms: float, resource_id: str) -> Optional[Alert]:
        """Check if latency exceeds threshold"""
        if current_latency_ms > self.latency_threshold_ms:
            alert = Alert(
                alert_type=AlertType.LATENCY_THRESHOLD,
                severity="critical" if current_latency_ms > self.latency_threshold_ms * 2 else "high",
                message=f"Latency {current_latency_ms:.1f}ms exceeds threshold {self.latency_threshold_ms}ms",
                current_value=current_latency_ms,
                threshold=self.latency_threshold_ms,
                resource_id=resource_id
            )
            self.alerts.append(alert)
            return alert
        return None

    def check_capacity_threshold(self, current_capacity_percent: float, resource_id: str) -> Optional[Alert]:
        """Check if capacity exceeds threshold"""
        if current_capacity_percent > self.capacity_threshold_percent:
            alert = Alert(
                alert_type=AlertType.CAPACITY_WARNING,
                severity="critical" if current_capacity_percent > 95 else "high",
                message=f"Capacity at {current_capacity_percent:.1f}% exceeds threshold {self.capacity_threshold_percent}%",
                current_value=current_capacity_percent,
                threshold=self.capacity_threshold_percent,
                resource_id=resource_id
            )
            self.alerts.append(alert)
            return alert
        return None

    def check_performance_degradation(self, current_throughput: float,
                                     baseline_throughput: float, resource_id: str) -> Optional[Alert]:
        """Check if performance has degraded significantly"""
        degradation_percent = ((baseline_throughput - current_throughput) / baseline_throughput) * 100
        if degradation_percent > 30:
            alert = Alert(
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                severity="high",
                message=f"Performance degraded by {degradation_percent:.1f}% from baseline",
                current_value=current_throughput,
                threshold=baseline_throughput * 0.7,
                resource_id=resource_id
            )
            self.alerts.append(alert)
            return alert
        return None

    def get_active_alerts(self, severity_filter: str = None) -> List[Alert]:
        """Get all active alerts, optionally filtered by severity"""
        active = [a for a in self.alerts if not a.acknowledged]
        if severity_filter:
            active = [a for a in active if a.severity == severity_filter]
        return active

    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                break


def apply_security_policies(files: List[Dict]) -> Dict:
    """Apply security policies to all files"""
    policies = []

    for file in files:
        file_id = file.get('file_id', 'UNKNOWN')
        location = file.get('location', 'on-premise')

        # Determine classification based on file characteristics
        size_mb = file.get('size_mb', 0)
        name = file.get('name', '').lower()

        if 'confidential' in name or 'secret' in name:
            classification = "confidential"
        elif 'financial' in name or 'employee' in name or 'customer' in name:
            classification = "sensitive"
        elif size_mb > 5000:
            classification = "sensitive"
        elif 'public' in name:
            classification = "public"
        else:
            classification = "internal"

        policy = SecurityPolicy(file_id, location, classification)
        policies.append(policy.to_dict())

    return {
        'total_files': len(files),
        'policies_applied': len(policies),
        'encryption_summary': {
            'military_grade': sum(1 for p in policies if p['encryption_level'] == 'military'),
            'enhanced': sum(1 for p in policies if p['encryption_level'] == 'enhanced'),
            'standard': sum(1 for p in policies if p['encryption_level'] == 'standard'),
            'none': sum(1 for p in policies if p['encryption_level'] == 'none')
        },
        'access_policy_summary': {
            'confidential': sum(1 for p in policies if p['access_policy'] == 'confidential'),
            'restricted': sum(1 for p in policies if p['access_policy'] == 'restricted'),
            'private': sum(1 for p in policies if p['access_policy'] == 'private'),
            'public': sum(1 for p in policies if p['access_policy'] == 'public')
        },
        'policies': policies
    }


def check_system_alerts(metrics: Dict) -> Dict:
    """Check all system metrics and generate alerts"""
    engine = AlertingEngine()

    # Check cost thresholds
    monthly_cost = metrics.get('monthly_cost', 0)
    if monthly_cost > 0:
        engine.check_cost_threshold(monthly_cost, "system_wide")

    # Check latency thresholds
    avg_latency = metrics.get('avg_latency_ms', 0)
    if avg_latency > 0:
        engine.check_latency_threshold(avg_latency, "system_wide")

    # Check capacity thresholds
    storage_used_percent = metrics.get('storage_used_percent', 0)
    if storage_used_percent > 0:
        engine.check_capacity_threshold(storage_used_percent, "storage")

    # Check performance
    current_throughput = metrics.get('current_throughput', 0)
    baseline_throughput = metrics.get('baseline_throughput', 100)
    if current_throughput > 0:
        engine.check_performance_degradation(current_throughput, baseline_throughput, "system_wide")

    active_alerts = engine.get_active_alerts()
    critical_alerts = [a for a in active_alerts if a.severity == "critical"]
    high_alerts = [a for a in active_alerts if a.severity == "high"]

    return {
        'timestamp': datetime.now().isoformat(),
        'total_alerts': len(active_alerts),
        'critical_alerts': len(critical_alerts),
        'high_priority_alerts': len(high_alerts),
        'alerts': [a.to_dict() for a in active_alerts],
        'thresholds': {
            'cost_monthly': engine.cost_threshold_monthly,
            'latency_ms': engine.latency_threshold_ms,
            'capacity_percent': engine.capacity_threshold_percent,
            'error_rate_percent': engine.error_rate_threshold_percent
        },
        'alert_summary': {
            'cost_violations': sum(1 for a in active_alerts if a.alert_type == AlertType.COST_THRESHOLD),
            'latency_violations': sum(1 for a in active_alerts if a.alert_type == AlertType.LATENCY_THRESHOLD),
            'capacity_warnings': sum(1 for a in active_alerts if a.alert_type == AlertType.CAPACITY_WARNING),
            'performance_issues': sum(1 for a in active_alerts if a.alert_type == AlertType.PERFORMANCE_DEGRADATION)
        }
    }


def export_report_json(report: Dict, filename: str):
    """Export report to JSON file"""
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report exported to {filename}")


if __name__ == "__main__":
    print("=" * 80)
    print("COMPONENT 6: SECURITY, ENCRYPTION & AUTOMATED ALERTS")
    print("=" * 80)

    # Sample files for demonstration
    sample_files = [
        {'file_id': 'FILE001', 'name': 'customer_data.db', 'size_mb': 2500, 'location': 'aws'},
        {'file_id': 'FILE002', 'name': 'financial_report_confidential.xlsx', 'size_mb': 50, 'location': 'on-premise'},
        {'file_id': 'FILE003', 'name': 'public_website_assets.zip', 'size_mb': 1000, 'location': 'gcp'},
        {'file_id': 'FILE004', 'name': 'employee_records.csv', 'size_mb': 500, 'location': 'azure'},
        {'file_id': 'FILE005', 'name': 'application_logs.tar.gz', 'size_mb': 8000, 'location': 'private-cloud'},
    ]

    # Apply security policies
    print("\n1. Applying Security Policies & Encryption...")
    security_report = apply_security_policies(sample_files)
    print(f"   Policies applied to {security_report['total_files']} files")
    print(f"   Encryption: {security_report['encryption_summary']}")
    print(f"   Access Control: {security_report['access_policy_summary']}")

    # Check for alerts
    print("\n2. Checking System Alerts...")
    sample_metrics = {
        'monthly_cost': 125.50,
        'avg_latency_ms': 650,
        'storage_used_percent': 88,
        'current_throughput': 45,
        'baseline_throughput': 100
    }

    alert_report = check_system_alerts(sample_metrics)
    print(f"   Total Alerts: {alert_report['total_alerts']}")
    print(f"   Critical: {alert_report['critical_alerts']}")
    print(f"   High Priority: {alert_report['high_priority_alerts']}")

    if alert_report['alerts']:
        print("\n   Active Alerts:")
        for alert in alert_report['alerts'][:5]:
            print(f"   - [{alert['severity'].upper()}] {alert['message']}")

    # Export comprehensive report
    comprehensive_report = {
        'timestamp': datetime.now().isoformat(),
        'security': security_report,
        'alerts': alert_report
    }

    export_report_json(comprehensive_report, 'data/exports/security_alerts_report.json')

    print("\n" + "=" * 80)
    print("COMPONENT 6 COMPLETE")
    print("=" * 80)

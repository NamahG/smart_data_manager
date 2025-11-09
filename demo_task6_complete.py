#!/usr/bin/env python3
"""
Demo Task 6: Security, Encryption & Automated Alerts
Complete demonstration of adaptive security policies and threshold-based alerting
"""

from component6_security_alerts import (
    apply_security_policies, check_system_alerts,
    export_report_json, SecurityPolicy
)
from datetime import datetime
import json


def create_sample_dataset():
    """Create sample files representing various data types"""
    return [
        # Confidential files
        {'file_id': 'CONF001', 'name': 'trade_secrets_confidential.pdf', 'size_mb': 150, 'location': 'on-premise'},
        {'file_id': 'CONF002', 'name': 'executive_salary_confidential.xlsx', 'size_mb': 25, 'location': 'on-premise'},

        # Sensitive files
        {'file_id': 'SENS001', 'name': 'customer_database.db', 'size_mb': 3500, 'location': 'aws'},
        {'file_id': 'SENS002', 'name': 'financial_records_2024.csv', 'size_mb': 500, 'location': 'azure'},
        {'file_id': 'SENS003', 'name': 'employee_personal_info.json', 'size_mb': 200, 'location': 'private-cloud'},
        {'file_id': 'SENS004', 'name': 'patient_medical_records.db', 'size_mb': 7500, 'location': 'on-premise'},

        # Internal files
        {'file_id': 'INT001', 'name': 'internal_meeting_notes.docx', 'size_mb': 10, 'location': 'gcp'},
        {'file_id': 'INT002', 'name': 'project_documentation.pdf', 'size_mb': 450, 'location': 'aws'},
        {'file_id': 'INT003', 'name': 'code_repository.tar.gz', 'size_mb': 2000, 'location': 'private-cloud'},

        # Public files
        {'file_id': 'PUB001', 'name': 'public_website_content.zip', 'size_mb': 800, 'location': 'gcp'},
        {'file_id': 'PUB002', 'name': 'public_api_documentation.html', 'size_mb': 5, 'location': 'aws'},
        {'file_id': 'PUB003', 'name': 'public_marketing_materials.pdf', 'size_mb': 150, 'location': 'azure'},
    ]


def simulate_system_metrics():
    """Simulate system metrics that may trigger alerts"""
    return {
        'monthly_cost': 156.75,
        'avg_latency_ms': 725,
        'storage_used_percent': 91,
        'current_throughput': 42,
        'baseline_throughput': 100,
        'error_rate_percent': 2.3
    }


def main():
    print("=" * 80)
    print("TASK 6: SECURITY, ENCRYPTION & AUTOMATED ALERTS DEMONSTRATION")
    print("=" * 80)

    # Create sample dataset
    files = create_sample_dataset()
    print(f"\nCreated sample dataset with {len(files)} files")
    print("File types: Confidential, Sensitive, Internal, Public")

    # Part 1: Apply Security Policies
    print("\n" + "=" * 80)
    print("PART 1: ADAPTIVE SECURITY POLICIES & ENCRYPTION")
    print("=" * 80)

    security_report = apply_security_policies(files)

    print(f"\nSecurity Analysis Complete:")
    print(f"  Total Files Analyzed: {security_report['total_files']}")
    print(f"  Policies Applied: {security_report['policies_applied']}")

    print(f"\nEncryption Summary:")
    enc_summary = security_report['encryption_summary']
    print(f"  Military Grade (AES-256 Multi-layer): {enc_summary.get('military', 0)} files")
    print(f"  Enhanced (AES-256): {enc_summary.get('enhanced', 0)} files")
    print(f"  Standard (AES-128): {enc_summary.get('standard', 0)} files")
    print(f"  No Encryption: {enc_summary.get('none', 0)} files")

    print(f"\nAccess Control Summary:")
    access_summary = security_report['access_policy_summary']
    print(f"  Confidential (On-premise only): {access_summary['confidential']} files")
    print(f"  Restricted (On-premise + Private cloud): {access_summary['restricted']} files")
    print(f"  Private (Multi-cloud allowed): {access_summary['private']} files")
    print(f"  Public (Unrestricted): {access_summary['public']} files")

    # Show examples
    print("\nExample Security Policies:")
    for policy in security_report['policies'][:3]:
        print(f"\n  File: {policy['file_id']}")
        print(f"    Classification: {policy['data_classification']}")
        print(f"    Encryption: {policy['encryption_level']}")
        print(f"    Access Policy: {policy['access_policy']}")
        print(f"    Key ID: {policy['encryption_key_id']}")
        print(f"    Allowed Regions: {', '.join(policy['allowed_regions'])}")

    # Part 2: Automated Alerts
    print("\n" + "=" * 80)
    print("PART 2: AUTOMATED THRESHOLD-BASED ALERTS")
    print("=" * 80)

    metrics = simulate_system_metrics()
    print(f"\nCurrent System Metrics:")
    print(f"  Monthly Cost: ${metrics['monthly_cost']:.2f}")
    print(f"  Average Latency: {metrics['avg_latency_ms']:.1f} ms")
    print(f"  Storage Usage: {metrics['storage_used_percent']:.1f}%")
    print(f"  Throughput: {metrics['current_throughput']}/{metrics['baseline_throughput']} ops/sec")

    alert_report = check_system_alerts(metrics)

    print(f"\nAlert Analysis:")
    print(f"  Total Active Alerts: {alert_report['total_alerts']}")
    print(f"  Critical Alerts: {alert_report['critical_alerts']}")
    print(f"  High Priority: {alert_report['high_priority_alerts']}")

    print(f"\nConfigured Thresholds:")
    thresholds = alert_report['thresholds']
    print(f"  Cost Threshold: ${thresholds['cost_monthly']:.2f}/month")
    print(f"  Latency Threshold: {thresholds['latency_ms']} ms")
    print(f"  Capacity Threshold: {thresholds['capacity_percent']}%")
    print(f"  Error Rate Threshold: {thresholds['error_rate_percent']}%")

    print(f"\nAlert Summary by Type:")
    summary = alert_report['alert_summary']
    print(f"  Cost Violations: {summary['cost_violations']}")
    print(f"  Latency Violations: {summary['latency_violations']}")
    print(f"  Capacity Warnings: {summary['capacity_warnings']}")
    print(f"  Performance Issues: {summary['performance_issues']}")

    if alert_report['alerts']:
        print(f"\nActive Alerts (showing all {len(alert_report['alerts'])}):")
        for i, alert in enumerate(alert_report['alerts'], 1):
            severity_marker = {
                'critical': '[CRITICAL]',
                'high': '[HIGH]',
                'medium': '[MEDIUM]',
                'low': '[LOW]'
            }.get(alert['severity'], '[INFO]')
            print(f"\n  Alert {i}: {severity_marker}")
            print(f"    Type: {alert['alert_type']}")
            print(f"    Message: {alert['message']}")
            print(f"    Current Value: {alert['current_value']:.2f}")
            print(f"    Threshold: {alert['threshold']:.2f}")
            print(f"    Timestamp: {alert['timestamp']}")

    # Part 3: Integration
    print("\n" + "=" * 80)
    print("PART 3: INTEGRATED SECURITY & ALERTING REPORT")
    print("=" * 80)

    # Create comprehensive report
    comprehensive_report = {
        'task': 'Task 6: Security, Encryption & Automated Alerts',
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'files_analyzed': security_report['total_files'],
            'policies_applied': security_report['policies_applied'],
            'encrypted_files': security_report['total_files'] - enc_summary['none'],
            'total_alerts': alert_report['total_alerts'],
            'critical_alerts': alert_report['critical_alerts']
        },
        'security': security_report,
        'alerts': alert_report,
        'recommendations': generate_recommendations(security_report, alert_report)
    }

    # Export reports
    export_report_json(comprehensive_report, 'data/exports/task6_security_alerts.json')
    export_report_json(security_report, 'data/exports/security_policies.json')
    export_report_json(alert_report, 'data/exports/system_alerts.json')

    # Summary
    print(f"\nSummary:")
    print(f"  Files Secured: {comprehensive_report['summary']['encrypted_files']}/{comprehensive_report['summary']['files_analyzed']}")
    print(f"  Active Alerts: {comprehensive_report['summary']['total_alerts']}")
    print(f"  Critical Issues: {comprehensive_report['summary']['critical_alerts']}")

    print(f"\nReports exported:")
    print(f"  - data/exports/task6_security_alerts.json")
    print(f"  - data/exports/security_policies.json")
    print(f"  - data/exports/system_alerts.json")

    print("\n" + "=" * 80)
    print("TASK 6 COMPLETE")
    print("=" * 80)


def generate_recommendations(security_report, alert_report):
    """Generate actionable recommendations"""
    recommendations = []

    # Security recommendations
    if security_report['encryption_summary']['none'] > 0:
        recommendations.append({
            'type': 'security',
            'priority': 'high',
            'recommendation': f"Enable encryption for {security_report['encryption_summary']['none']} unencrypted files",
            'action': 'Apply at least standard AES-128 encryption to all files'
        })

    # Alert-based recommendations
    if alert_report['critical_alerts'] > 0:
        recommendations.append({
            'type': 'operational',
            'priority': 'critical',
            'recommendation': f"Address {alert_report['critical_alerts']} critical alerts immediately",
            'action': 'Review and resolve critical threshold violations'
        })

    if alert_report['alert_summary']['cost_violations'] > 0:
        recommendations.append({
            'type': 'cost',
            'priority': 'high',
            'recommendation': 'Reduce monthly costs through optimization',
            'action': 'Review data placement and migrate cold data to cheaper tiers'
        })

    if alert_report['alert_summary']['latency_violations'] > 0:
        recommendations.append({
            'type': 'performance',
            'priority': 'high',
            'recommendation': 'Improve system latency',
            'action': 'Move frequently accessed data closer to compute resources'
        })

    return recommendations


if __name__ == "__main__":
    main()

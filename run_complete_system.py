#!/usr/bin/env python3
"""
Intelligent Data Fabric - Complete System Runner
Runs all components and launches dashboard
"""

import subprocess
import sys
import time
from pathlib import Path

def print_header(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f">> {description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"[OK] {description} complete\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed: {e}\n")
        return False

def main():
    print_header("INTELLIGENT DATA FABRIC - COMPLETE SYSTEM")

    print("This will run all 7 components in sequence:\n")
    print("  1. Data Placement Optimization")
    print("  2. Multi-Cloud Migration")
    print("  3. Real-Time Streaming")
    print("  4. Predictive ML Insights")
    print("  5. Consistency & Availability")
    print("  6. Security & Automated Alerts")
    print("  7. Interactive Dashboard\n")

    input("Press Enter to start...")

    # Run all demo scripts
    components = [
        ("python3 demo_task1_complete.py", "Task 1: Data Placement Optimization"),
        ("python3 demo_task2_complete.py", "Task 2: Multi-Cloud Migration"),
        ("python3 demo_task3_complete.py", "Task 3: Real-Time Streaming"),
        ("python3 demo_task4_complete.py", "Task 4: Predictive ML"),
        ("python3 demo_task5_complete.py", "Task 5: Consistency & Availability"),
        ("python3 demo_task6_complete.py", "Task 6: Security & Alerts"),
    ]

    print_header("RUNNING PIPELINE")

    success_count = 0
    for cmd, desc in components:
        if run_command(cmd, desc):
            success_count += 1
        time.sleep(1)

    # Summary
    print_header("PIPELINE EXECUTION COMPLETE")
    print(f"  Successfully completed: {success_count}/{len(components)} components\n")

    if success_count == len(components):
        print("ALL COMPONENTS RAN SUCCESSFULLY!\n")
        print("Now launching the interactive dashboard...\n")
        print("="*80)
        print("\n  Dashboard will auto-discover a free port")
        print("  Watch the output below for the exact URL")
        print("  Press Ctrl+C to stop the dashboard\n")
        print("="*80 + "\n")

        time.sleep(2)

        # Launch dashboard
        try:
            subprocess.run(["python3", "web_dashboard.py"], check=True)
        except KeyboardInterrupt:
            print("\n\n Dashboard stopped.")
        except Exception as e:
            print(f"\n\n[ERROR] Dashboard error: {e}")
            print("\nYou can manually launch it with: python3 web_dashboard.py")
    else:
        print("WARNING: Some components failed. Check the errors above.")
        print("         You can still try running the dashboard: python3 web_dashboard.py")

if __name__ == "__main__":
    main()

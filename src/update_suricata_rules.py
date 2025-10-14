#!/usr/bin/env python3
"""
Suricata Rule Updater Script

This script updates Suricata rules and restarts the Suricata service.
It can be run manually or scheduled via Windows Task Scheduler.
"""

import subprocess
import logging
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('suricata_update.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_command(command: str) -> tuple[bool, str]:
    """Run a shell command and return success status and output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Command failed: {e.stderr}"

def update_suricata_rules() -> bool:
    """Update Suricata rules and restart the service."""
    logging.info("Starting Suricata rule update...")
    
    # Update rules
    success, output = run_command("sudo suricata-update")
    if not success:
        logging.error(f"Failed to update rules: {output}")
        return False
    
    logging.info("Rules updated successfully")
    
    # Restart Suricata service
    success, output = run_command("sudo systemctl restart suricata")
    if not success:
        logging.error(f"Failed to restart Suricata: {output}")
        return False
    
    logging.info("Suricata service restarted successfully")
    return True

def main():
    logging.info("=" * 50)
    logging.info(f"Starting Suricata rule update at {datetime.now()}")
    
    if update_suricata_rules():
        logging.info("Suricata rule update completed successfully")
    else:
        logging.error("Suricata rule update failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
